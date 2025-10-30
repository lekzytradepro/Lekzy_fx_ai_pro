import os
import asyncio
import sqlite3
import json
import time
import random
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask
from threading import Thread

# ==================== CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "LEKZY_ADMIN_123")  # Change this!
    DB_PATH = "/app/data/lekzy_fx_ai.db"
    PORT = int(os.getenv("PORT", 10000))

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LEKZY_FX_AI")

# ==================== WEB SERVER FOR RENDER ====================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 LEKZY FX AI PRO - Trading Bot Active 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: Healthy"

def run_web_server():
    app.run(host='0.0.0.0', port=Config.PORT)

def start_web_server():
    web_thread = Thread(target=run_web_server)
    web_thread.daemon = True
    web_thread.start()
    logger.info(f"🌐 Web server started on port {Config.PORT}")

# ==================== DATABASE SETUP ====================
def initialize_database():
    """Initialize all database tables"""
    try:
        os.makedirs("/app/data", exist_ok=True)
        
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                login_time TEXT,
                expiry_time TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                plan_type TEXT DEFAULT 'TRIAL',
                start_date TEXT,
                end_date TEXT,
                payment_status TEXT DEFAULT 'PENDING',
                signals_used INTEGER DEFAULT 0,
                max_daily_signals INTEGER DEFAULT 5,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notification_type TEXT,
                user_id INTEGER,
                username TEXT,
                sent_time TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                signal_type TEXT,
                direction TEXT,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                confidence REAL,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")

# ==================== ADMIN AUTHENTICATION ====================
class AdminAuth:
    def __init__(self):
        self.session_duration = timedelta(hours=24)
    
    def verify_token(self, token: str) -> bool:
        """Verify admin token"""
        return token == Config.ADMIN_TOKEN
    
    def create_session(self, user_id: int, username: str):
        """Create admin session"""
        login_time = datetime.now()
        expiry_time = login_time + self.session_duration
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO admin_sessions 
                (user_id, username, login_time, expiry_time)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, login_time.isoformat(), expiry_time.isoformat()))
            conn.commit()
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user has active admin session"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT expiry_time FROM admin_sessions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                expiry_time = datetime.fromisoformat(result[0])
                if expiry_time > datetime.now():
                    return True
                else:
                    conn.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
                    conn.commit()
            return False

# ==================== TRADING SESSIONS ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {"start_hour": 8, "end_hour": 12, "name": "European", "pairs": ["EUR/USD", "GBP/USD"]},
            "EVENING": {"start_hour": 16, "end_hour": 20, "name": "NY/London", "pairs": ["USD/JPY", "XAU/USD"]},
            "ASIAN": {"start_hour": 0, "end_hour": 4, "name": "Asian", "pairs": ["AUD/JPY", "USD/JPY"]}
        }

    def get_current_session(self):
        """Get current active trading session"""
        now = datetime.now()
        current_hour = now.hour
        
        for session_id, session in self.sessions.items():
            if session["start_hour"] <= current_hour < session["end_hour"]:
                return {**session, "id": session_id}
        
        return {"id": "CLOSED", "name": "Market Closed"}

# ==================== MINIMAL SIGNAL GENERATOR ====================
class MinimalSignalGenerator:
    def generate_signal(self, symbol: str):
        """Generate minimal trading signal"""
        direction = random.choice(["BUY", "SELL"])
        base_price = random.uniform(1.0500, 1.1500) if "EUR" in symbol else random.uniform(1.2500, 1.3500)
        
        if direction == "BUY":
            entry = round(base_price, 5)
            tp = round(entry * 1.004, 5)
            sl = round(entry * 0.998, 5)
        else:
            entry = round(base_price, 5)
            tp = round(entry * 0.996, 5)
            sl = round(entry * 1.002, 5)
        
        signal_id = f"SIG_{int(time.time())}_{random.randint(1000,9999)}"
        
        return {
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "take_profit": tp,
            "stop_loss": sl,
            "confidence": round(random.uniform(0.85, 0.97), 2)
        }

# ==================== NOTIFICATION MANAGER ====================
class NotificationManager:
    def __init__(self, application):
        self.application = application
        self.admin_auth = AdminAuth()
    
    async def notify_new_user(self, user_id: int, username: str):
        """Notify all active admins about new user"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("SELECT user_id FROM admin_sessions")
            active_admins = [row[0] for row in cursor.fetchall()]
        
        if not active_admins:
            return
            
        message = f"""
👤 NEW USER: @{username}
🆔 ID: {user_id}
⏰ {datetime.now().strftime('%H:%M')}
"""
        
        for admin_id in active_admins:
            try:
                await self.application.bot.send_message(chat_id=admin_id, text=message)
            except Exception:
                pass

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str, notification_manager: NotificationManager):
        self.db_path = db_path
        self.notification_manager = notification_manager
    
    def start_trial(self, user_id: int, username: str):
        """Start free trial"""
        end_date = datetime.now() + timedelta(days=3)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_type, start_date, end_date, payment_status)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, "TRIAL", datetime.now().isoformat(), end_date.isoformat(), "ACTIVE"))
            conn.commit()
        
        asyncio.create_task(self.notification_manager.notify_new_user(user_id, username))

# ==================== TRADING BOT ====================
class TradingBot:
    def __init__(self):
        self.session_manager = SessionManager()
        self.signal_generator = MinimalSignalGenerator()
        self.is_running = False
    
    async def start_signals(self):
        """Start signal generation"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED":
                        for symbol in session["pairs"][:1]:
                            signal = self.signal_generator.generate_signal(symbol)
                            
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, direction, entry_price, take_profit, stop_loss, confidence)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    signal["signal_id"], signal["symbol"], signal["direction"],
                                    signal["entry_price"], signal["take_profit"], 
                                    signal["stop_loss"], signal["confidence"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📡 Signal: {signal['symbol']} {signal['direction']}")
                            
                    await asyncio.sleep(random.randint(300, 600))
                    
                except Exception as e:
                    logger.error(f"Signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())

# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.notification_manager = None
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the bot"""
        self.application = Application.builder().token(self.token).build()
        self.notification_manager = NotificationManager(self.application)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH, self.notification_manager)
        self.trading_bot = TradingBot()
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_signals()
        
        logger.info("🤖 Bot initialized")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username)
        
        message = f"""
🎯 Welcome {user.first_name}!

3-Day Trial Activated
Signals: 5/day | Session: Morning

Commands:
• /stats - Account
• /session - Market hours  
• /signals - Recent trades
• /login - Admin access
"""
        await update.message.reply_text(message)

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command"""
        user = update.effective_user
        
        if not context.args:
            await update.message.reply_text("🔐 Usage: /login <admin_token>")
            return
        
        token = context.args[0]
        
        if self.admin_auth.verify_token(token):
            self.admin_auth.create_session(user.id, user.username)
            await update.message.reply_text("✅ Admin access granted")
            await self.admin_command(update, context)
        else:
            await update.message.reply_text("❌ Invalid token")

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required\nUse: /login <token>")
            return
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        
        message = f"""
🏢 ADMIN PANEL

Users: {users}
Signals: {signals}
Session: {self.trading_bot.session_manager.get_current_session()['name']}

Commands:
• /stats - System stats
• /logout - End session
"""
        await update.message.reply_text(message)

    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout command"""
        user = update.effective_user
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user.id,))
            conn.commit()
        
        await update.message.reply_text("✅ Logged out")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            plan = conn.execute("SELECT plan_type FROM subscriptions WHERE user_id = ?", (user.id,)).fetchone()
            user_plan = plan[0] if plan else "TRIAL"
            
            if self.admin_auth.is_admin(user.id):
                users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
                signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
                message = f"""
📊 SYSTEM STATS

Users: {users}
Signals: {signals}
Plan: {user_plan}
Status: ✅ Active
"""
            else:
                message = f"""
📊 ACCOUNT STATS

Plan: {user_plan}
Signals: 5/day
Session: Morning
Status: ✅ Active
"""
        
        await update.message.reply_text(message)

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
        session = self.trading_bot.session_manager.get_current_session()
        
        if session["id"] == "CLOSED":
            message = """
🕒 MARKET CLOSED

Next Sessions:
• Morning: 08:00-12:00
• Evening: 16:00-20:00
• Asian: 00:00-04:00
"""
        else:
            message = f"""
🕒 {session['name']} SESSION

Status: ✅ ACTIVE
Pairs: {', '.join(session['pairs'])}
Signals: ✅ Active
"""
        
        await update.message.reply_text(message)

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, direction, entry_price, confidence, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet")
            return
        
        message = "📡 RECENT SIGNALS\n\n"
        
        for symbol, direction, entry, confidence, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            arrow = "🟢" if direction == "BUY" else "🔴"
            message += f"{arrow} {symbol} {direction}\n"
            message += f"💵 {entry} | {confidence*100:.1f}%\n"
            message += f"⏰ {time_str}\n\n"
        
        await update.message.reply_text(message)

    async def start_polling(self):
        """Start polling"""
        await self.application.updater.start_polling()

    async def stop(self):
        """Stop bot"""
        self.trading_bot.is_running = False
        await self.application.stop()

# ==================== MAIN APPLICATION ====================
class MainApp:
    def __init__(self):
        self.bot = None
        self.running = False
    
    async def setup(self):
        """Setup application"""
        initialize_database()
        start_web_server()
        
        self.bot = TelegramBot()
        await self.bot.initialize()
        
        self.running = True
        logger.info("🚀 LEKZY FX AI PRO Started")
    
    async def run(self):
        """Run application"""
        if not self.running:
            await self.setup()
        
        await self.bot.start_polling()
        
        while self.running:
            await asyncio.sleep(10)
    
    async def shutdown(self):
        """Shutdown"""
        self.running = False
        if self.bot:
            await self.bot.stop()

# ==================== START BOT ====================
async def main():
    app = MainApp()
    try:
        await app.run()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
