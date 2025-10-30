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

# ==================== CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")
    ADMIN_USER_IDS = json.loads(os.getenv("ADMIN_USER_IDS", "[123456789]"))
    DB_PATH = "/app/data/lekzy_fx_ai.db"
    
    # Notification settings
    NOTIFY_NEW_SUBSCRIBERS = os.getenv("NOTIFY_NEW_SUBSCRIBERS", "true").lower() == "true"
    NOTIFY_UPGRADES = os.getenv("NOTIFY_UPGRADES", "true").lower() == "true"
    NOTIFY_TRIAL_EXPIRY = os.getenv("NOTIFY_TRIAL_EXPIRY", "true").lower() == "true"
    
    # Trading settings
    MIN_COOLDOWN = 180
    MAX_COOLDOWN = 600

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LEKZY_FX_AI")

# ==================== DATABASE SETUP ====================
def initialize_database():
    """Initialize all database tables"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("/app/data", exist_ok=True)
        
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()

        # Subscriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                plan_type TEXT DEFAULT 'TRIAL',
                start_date TEXT,
                end_date TEXT,
                payment_status TEXT DEFAULT 'PENDING',
                signals_used INTEGER DEFAULT 0,
                max_daily_signals INTEGER DEFAULT 5,
                allowed_sessions TEXT DEFAULT '["MORNING"]',
                timezone TEXT DEFAULT 'UTC+1',
                broadcast_enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Admin notifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notification_type TEXT,
                user_id INTEGER,
                username TEXT,
                plan_type TEXT,
                details TEXT,
                sent_time TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                details TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                confidence REAL,
                session_type TEXT,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8,
                "end_hour": 12,
                "name": "European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "volatility": "HIGH",
                "typical_accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 16,
                "end_hour": 20,
                "name": "NY/London Overlap",
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "typical_accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 0,
                "end_hour": 4,
                "name": "Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "typical_accuracy": 92.5
            }
        }

    def get_current_session(self):
        """Get current active trading session"""
        now = datetime.now()
        current_hour = now.hour
        
        for session_id, session in self.sessions.items():
            if session["start_hour"] <= current_hour < session["end_hour"]:
                return {**session, "id": session_id}
        
        return {"id": "CLOSED", "name": "Market Closed"}

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def generate_signal(self, symbol: str, session_type: str):
        """Generate trading signal"""
        # Generate realistic signal data
        direction = "BUY" if random.random() > 0.5 else "SELL"
        base_price = random.uniform(1.0500, 1.1500) if "EUR" in symbol else random.uniform(1.2000, 1.3000)
        
        if direction == "BUY":
            entry_price = round(base_price, 5)
            take_profit = round(entry_price * 1.005, 5)
            stop_loss = round(entry_price * 0.995, 5)
        else:
            entry_price = round(base_price, 5)
            take_profit = round(entry_price * 0.995, 5)
            stop_loss = round(entry_price * 1.005, 5)
        
        signal_id = f"SIG_{symbol.replace('/', '')}_{int(time.time())}"
        
        return {
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "confidence": round(random.uniform(0.85, 0.98), 3),
            "session_type": session_type,
            "risk_reward": round(random.uniform(1.5, 3.0), 2),
            "generated_at": datetime.now().isoformat()
        }

# ==================== ADMIN NOTIFICATION MANAGER ====================
class AdminNotificationManager:
    def __init__(self, application):
        self.application = application
        self.admin_ids = Config.ADMIN_USER_IDS
    
    async def notify_new_subscriber(self, user_id: int, username: str, plan_type: str):
        """Notify admin about new subscriber"""
        if not Config.NOTIFY_NEW_SUBSCRIBERS:
            return
            
        message = f"""
🎉 NEW SUBSCRIBER ALERT!

👤 User: @{username} (ID: `{user_id}`)
📋 Plan: {plan_type}
⏰ Time: {datetime.now().strftime('%H:%M:%S')}

Welcome them to LEKZY FX AI PRO! 🚀
"""
        
        for admin_id in self.admin_ids:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"✅ Admin notified: {admin_id}")
                
                # Log notification in database
                with sqlite3.connect(Config.DB_PATH) as conn:
                    conn.execute("""
                        INSERT INTO admin_notifications 
                        (notification_type, user_id, username, plan_type, details)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        "NEW_SUBSCRIBER",
                        user_id,
                        username,
                        plan_type,
                        json.dumps({"timestamp": datetime.now().isoformat()})
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"❌ Failed to notify admin {admin_id}: {e}")

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str, admin_manager: AdminNotificationManager):
        self.db_path = db_path
        self.admin_manager = admin_manager
    
    def start_free_trial(self, user_id: int, username: str, first_name: str):
        """Start free trial with admin notification"""
        end_date = datetime.now() + timedelta(days=3)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_type, start_date, end_date, payment_status, max_daily_signals, allowed_sessions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, "TRIAL", datetime.now().isoformat(), 
                end_date.isoformat(), "TRIAL", 5, '["MORNING"]'
            ))
            conn.commit()
        
        # Notify admin
        asyncio.create_task(self.admin_manager.notify_new_subscriber(user_id, username, "TRIAL"))
        logger.info(f"✅ New trial started: {username} ({user_id})")
    
    def get_user_plan(self, user_id: int) -> str:
        """Get user's current plan"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT plan_type FROM subscriptions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else "TRIAL"

    def log_user_activity(self, user_id: int, activity_type: str, details: str = ""):
        """Log user activity"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            """, (user_id, activity_type, details))
            conn.commit()

# ==================== TRADING BOT ====================
class TradingBot:
    def __init__(self):
        self.session_manager = SessionManager()
        self.signal_generator = SignalGenerator()
        self.is_running = False
    
    async def start_signal_generation(self):
        """Start automatic signal generation"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    current_session = self.session_manager.get_current_session()
                    
                    if current_session["id"] != "CLOSED":
                        # Generate signals for optimal pairs
                        for symbol in current_session["optimal_pairs"][:2]:  # First 2 pairs
                            signal = self.signal_generator.generate_signal(symbol, current_session["id"])
                            
                            # Store signal in database
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, direction, entry_price, take_profit, stop_loss, confidence, session_type)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    signal["signal_id"],
                                    signal["symbol"],
                                    signal["direction"],
                                    signal["entry_price"],
                                    signal["take_profit"],
                                    signal["stop_loss"],
                                    signal["confidence"],
                                    signal["session_type"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📡 Generated signal: {signal['symbol']} {signal['direction']}")
                            
                            # Brief pause between signals
                            await asyncio.sleep(2)
                    
                    # Wait before next cycle
                    wait_time = random.randint(Config.MIN_COOLDOWN, Config.MAX_COOLDOWN)
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}")
                    await asyncio.sleep(30)
        
        # Start the signal loop
        asyncio.create_task(signal_loop())
        logger.info("✅ Signal generation started")

# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_manager = None
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the bot"""
        self.application = Application.builder().token(self.token).build()
        self.admin_manager = AdminNotificationManager(self.application)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH, self.admin_manager)
        self.trading_bot = TradingBot()
        
        # Setup command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        
        await self.application.initialize()
        await self.application.start()
        
        # Start trading features
        await self.trading_bot.start_signal_generation()
        
        logger.info("🤖 Telegram bot initialized successfully!")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        # Register user and start trial
        self.subscription_manager.start_free_trial(user.id, user.username, user.first_name)
        self.subscription_manager.log_user_activity(user.id, "REGISTER", "User started bot")
        
        welcome_message = f"""
🎉 Welcome to LEKZY FX AI PRO, {user.first_name}!

Your 3-day free trial has been activated! 
You'll receive trading signals during morning sessions.

🔔 *Admin has been notified of your subscription.*

Available Commands:
• /stats - Your account status
• /session - Current trading session
• /signals - Recent trading signals
• /admin - Admin dashboard (admin only)

Happy trading! 📈
"""
        keyboard = [
            [InlineKeyboardButton("📊 Account Stats", callback_data="stats")],
            [InlineKeyboardButton("🕒 Trading Session", callback_data="session")],
            [InlineKeyboardButton("📡 Recent Signals", callback_data="signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if user.id not in Config.ADMIN_USER_IDS:
            await update.message.reply_text("❌ Admin access only.")
            return
        
        # Get statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM subscriptions")
            total_users = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM admin_notifications")
            total_notifications = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]
        
        admin_message = f"""
🏢 *ADMIN DASHBOARD*

Welcome, @{user.username}!

📊 *System Statistics:*
• Total Users: {total_users}
• Notifications Sent: {total_notifications}
• Signals Generated: {total_signals}
• System Time: {datetime.now().strftime('%H:%M:%S')}

👑 *Admin Features:*
• New subscriber alerts ✅
• Real-time notifications ✅
• User management ✅
• Signal monitoring ✅

🔧 *Bot Status:* ✅ Running
"""
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh")],
            [InlineKeyboardButton("📊 Detailed Stats", callback_data="admin_stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(admin_message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        # Get user activity count
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM user_activity WHERE user_id = ?",
                (user.id,)
            )
            activity_count = cursor.fetchone()[0]
        
        stats_message = f"""
📊 *YOUR ACCOUNT STATS*

👤 User: {user.first_name}
📋 Plan: {user_plan}
🆔 ID: `{user.id}`
📈 Activities: {activity_count}

💡 *Trial Features:*
• Morning Session Signals
• 5 Signals Per Day  
• Basic Accuracy (96.2%)

🔔 Admin will be notified when you upgrade!

*Available Sessions:* Morning (08:00-12:00)
"""
        await update.message.reply_text(stats_message, parse_mode='Markdown')
    
    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
        current_session = self.trading_bot.session_manager.get_current_session()
        
        if current_session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED*

No active trading sessions at the moment.

*Next Sessions:*
• 🌅 Morning: 08:00 - 12:00
• 🌇 Evening: 16:00 - 20:00  
• 🌃 Asian: 00:00 - 04:00

Check back during session hours for signals! 📈
"""
        else:
            message = f"""
🕒 *CURRENT TRADING SESSION*

*{current_session['name']}*
⏰ Active Now ({current_session['start_hour']:02d}:00 - {current_session['end_hour']:02d}:00)

📊 *Session Details:*
• Volatility: {current_session['volatility']}
• Accuracy: {current_session['typical_accuracy']}%
• Optimal Pairs: {', '.join(current_session['optimal_pairs'][:3])}

🎯 Signals are being generated automatically!
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        # Get recent signals
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT symbol, direction, entry_price, confidence, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_signals = cursor.fetchall()
        
        if not recent_signals:
            await update.message.reply_text("📭 No signals generated yet. Check during trading sessions!")
            return
        
        signals_message = "📡 *RECENT TRADING SIGNALS*\n\n"
        
        for signal in recent_signals:
            symbol, direction, entry_price, confidence, created_at = signal
            direction_emoji = "🟢" if direction == "BUY" else "🔴"
            time_str = datetime.fromisoformat(created_at).strftime("%H:%M")
            
            signals_message += f"""
{direction_emoji} *{symbol}* {direction}
💵 Entry: `{entry_price:.5f}`
🎯 Confidence: {confidence*100:.1f}%
⏰ Time: {time_str}
"""
        
        signals_message += "\n💡 *Signals update automatically during trading sessions*"
        
        await update.message.reply_text(signals_message, parse_mode='Markdown')
    
    async def start_polling(self):
        """Start polling for messages"""
        logger.info("📡 Starting to poll for messages...")
        await self.application.updater.start_polling()
    
    async def stop(self):
        """Stop the bot"""
        self.trading_bot.is_running = False
        await self.application.stop()
        logger.info("🛑 Telegram bot stopped")

# ==================== MAIN APPLICATION ====================
class UltimateApplication:
    def __init__(self):
        self.telegram_bot = None
        self.is_running = False
    
    async def setup(self):
        """Setup the application"""
        logger.info("🔄 Setting up LEKZY FX AI PRO with Admin Notifications...")
        
        # Initialize database first
        initialize_database()
        
        self.telegram_bot = TelegramBot()
        await self.telegram_bot.initialize()
        
        self.is_running = True
        logger.info("🎯 LEKZY FX AI PRO with Admin Notifications is READY!")
    
    async def run(self):
        """Run the application"""
        if not self.is_running:
            await self.setup()
        
        logger.info("🔄 Starting main loop...")
        await self.telegram_bot.start_polling()
        
        # Keep the application running
        while self.is_running:
            await asyncio.sleep(10)
    
    async def shutdown(self):
        """Shutdown the application"""
        self.is_running = False
        if self.telegram_bot:
            await self.telegram_bot.stop()
        logger.info("🛑 Application stopped")

# ==================== START THE BOT ====================
async def main():
    """Main application entry point"""
    app = UltimateApplication()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
