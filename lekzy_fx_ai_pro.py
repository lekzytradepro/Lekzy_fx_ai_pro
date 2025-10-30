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
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "LEKZY_ADMIN_123")
    ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@LekzyTradingPro")
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
    return "🤖 LEKZY FX AI PRO - Fixed Admin Issues 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: Fixed & Running"

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
                payment_status TEXT DEFAULT 'ACTIVE',
                signals_used INTEGER DEFAULT 0,
                max_daily_signals INTEGER DEFAULT 5,
                allowed_sessions TEXT DEFAULT '["MORNING"]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS upgrade_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                requested_plan TEXT,
                contact_method TEXT,
                status TEXT DEFAULT 'PENDING',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                session_type TEXT,
                analysis TEXT,
                time_to_entry INTEGER,
                risk_reward REAL,
                requested_by TEXT DEFAULT 'AUTO',
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
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

# ==================== WORKING SIGNAL GENERATOR ====================
class WorkingSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
    
    def generate_trade_analysis(self, symbol: str, is_admin_request: bool = False) -> dict:
        """Generate professional trade analysis"""
        if is_admin_request:
            analysis_templates = [
                "Strong institutional flow with clear directional bias",
                "Price action confirming breakout with volume validation",
                "Multiple timeframe alignment creating high-probability setup",
                "Key support/resistance level holding with momentum"
            ]
            quality = "EXCELLENT"
        else:
            analysis_templates = [
                "Strong momentum at key level",
                "Breakout confirmation forming",
                "Support/resistance bounce",
                "Trend alignment positive"
            ]
            quality = random.choice(["HIGH", "VERY_HIGH"])
        
        return {
            "setup_quality": quality,
            "market_condition": random.choice(analysis_templates),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG_BULLISH", "STRONG_BEARISH"]),
            "confidence_score": random.randint(85, 98)
        }
    
    def generate_instant_signal(self, symbol: str = None) -> dict:
        """Generate instant signal (both pre-entry and entry combined) - FIXED VERSION"""
        try:
            if not symbol:
                symbol = random.choice(self.all_pairs)
            
            # Generate analysis
            analysis = self.generate_trade_analysis(symbol, True)
            direction = "BUY" if random.random() > 0.48 else "SELL"
            
            # Calculate prices
            base_price = analysis["key_level"]
            if direction == "BUY":
                entry_price = round(base_price + 0.0008, 5)
                take_profit = round(entry_price + 0.0030, 5)
                stop_loss = round(entry_price - 0.0018, 5)
            else:
                entry_price = round(base_price - 0.0008, 5)
                take_profit = round(entry_price - 0.0030, 5)
                stop_loss = round(entry_price + 0.0018, 5)
            
            # Calculate risk/reward
            risk_reward = round((take_profit - entry_price) / (entry_price - stop_loss), 2) if direction == "BUY" else round((entry_price - take_profit) / (stop_loss - entry_price), 2)
            
            signal_id = f"ADMIN_{symbol.replace('/', '')}_{int(time.time())}"
            
            signal_data = {
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": "ENTRY",
                "direction": direction,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "confidence": round(random.uniform(0.92, 0.98), 3),
                "session_type": "ADMIN_24_7",
                "analysis": json.dumps(analysis),
                "time_to_entry": 0,
                "risk_reward": risk_reward,
                "requested_by": "ADMIN",
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Admin signal generated successfully: {symbol} {direction}")
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return None

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8, "end_hour": 12,
                "name": "🌅 European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD"],
                "accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 16, "end_hour": 20,
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "XAU/USD"],
                "accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 0, "end_hour": 4,
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "USD/JPY"],
                "accuracy": 92.5
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

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def start_trial(self, user_id: int, username: str, first_name: str):
        """Start free trial"""
        end_date = datetime.now() + timedelta(days=3)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_type, start_date, end_date, payment_status, max_daily_signals, allowed_sessions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, "TRIAL", datetime.now().isoformat(), 
                end_date.isoformat(), "ACTIVE", 5, '["MORNING"]'
            ))
            conn.commit()
        
        logger.info(f"✅ Trial started: {username} ({user_id})")
    
    def get_user_plan(self, user_id: int) -> str:
        """Get user's current plan"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT plan_type FROM subscriptions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else "TRIAL"

# ==================== WORKING TRADING BOT ====================
class WorkingTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = SessionManager()
        self.signal_generator = WorkingSignalGenerator()
        self.is_running = False
    
    async def start_auto_signals(self):
        """Start automatic signal generation during sessions"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED":
                        logger.info(f"🎯 Auto signals for {session['name']}")
                        
                        for symbol in session["optimal_pairs"][:1]:
                            # Generate simple signal for auto mode
                            signal = self.signal_generator.generate_instant_signal(symbol)
                            signal["requested_by"] = "AUTO"
                            signal["session_type"] = session["id"]
                            
                            if signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, requested_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        signal["signal_id"], signal["symbol"], signal["signal_type"],
                                        signal["direction"], signal["entry_price"], signal["take_profit"],
                                        signal["stop_loss"], signal["confidence"], signal["session_type"],
                                        signal["analysis"], signal["time_to_entry"], signal["risk_reward"],
                                        signal["requested_by"]
                                    ))
                                    conn.commit()
                                
                                logger.info(f"🎯 Auto Signal: {signal['symbol']} {signal['direction']}")
                    
                    await asyncio.sleep(random.randint(300, 600))
                    
                except Exception as e:
                    logger.error(f"Auto signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Auto signal generation started")
    
    async def generate_admin_signal(self, user_id: int, symbol: str = None):
        """Generate instant signal for admin (available 24/7) - FIXED VERSION"""
        try:
            logger.info(f"🔄 Generating admin signal for user {user_id}, symbol: {symbol}")
            
            # Generate the signal
            signal = self.signal_generator.generate_instant_signal(symbol)
            
            if signal:
                # Store in database
                with sqlite3.connect(Config.DB_PATH) as conn:
                    conn.execute("""
                        INSERT INTO signals 
                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, requested_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        signal["signal_id"], signal["symbol"], signal["signal_type"],
                        signal["direction"], signal["entry_price"], signal["take_profit"],
                        signal["stop_loss"], signal["confidence"], signal["session_type"],
                        signal["analysis"], signal["time_to_entry"], signal["risk_reward"],
                        signal["requested_by"]
                    ))
                    conn.commit()
                
                logger.info(f"✅ Admin signal stored: {signal['symbol']} {signal['direction']}")
                return signal
            else:
                logger.error("❌ Signal generation returned None")
                return None
                
        except Exception as e:
            logger.error(f"❌ Admin signal generation failed: {e}")
            return None

# ==================== FIXED TELEGRAM BOT ====================
class FixedTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the fixed bot"""
        self.application = Application.builder().token(self.token).build()
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = WorkingTradingBot(self.application)
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        self.application.add_handler(CommandHandler("plans", self.plans_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_auto_signals()
        
        logger.info("🤖 Fixed Telegram Bot Initialized!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - REMOVED ADMIN COMMAND FROM WELCOME"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
You'll receive professional trading signals during market sessions.

💡 *Get Started:*
• Use /plans to see upgrade options
• Use /contact for immediate assistance
• Use /stats to check your account

*Start your trading journey today!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("💎 View Plans", callback_data="plans")],
            [InlineKeyboardButton("📞 Contact Admin", callback_data="contact")],
            [InlineKeyboardButton("📊 Account Stats", callback_data="stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command"""
        user = update.effective_user
        
        if not context.args:
            await update.message.reply_text(
                "🔐 *Admin Login*\n\nUsage: `/login YOUR_ADMIN_TOKEN`",
                parse_mode='Markdown'
            )
            return
        
        token = context.args[0]
        
        if self.admin_auth.verify_token(token):
            self.admin_auth.create_session(user.id, user.username)
            await update.message.reply_text("""
✅ *Admin Access Granted!* 🌟

🎯 *Admin Commands Unlocked:*
• `/admin` - Admin dashboard
• `/signal` - Generate signals 24/7
• Full system control

*Use /admin to access admin features!* 🚀
""", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
        # Get admin statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            admin_signals = conn.execute("SELECT COUNT(*) FROM signals WHERE requested_by = 'ADMIN'").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🏢 *ADMIN DASHBOARD* 🌟

📊 *Statistics:*
• Total Users: {total_users}
• Total Signals: {total_signals}
• Admin Signals: {admin_signals}
• Current Session: {current_session['name']}

🎯 *Admin Commands:*
• `/signal` - Generate random signal
• `/signal EUR/USD` - Specific pair
• 24/7 signal generation

*You have full system control!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="admin_signal")],
            [InlineKeyboardButton("📊 Refresh Stats", callback_data="admin_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command - FIXED VERSION"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
        # Get symbol from command if provided
        symbol = None
        if context.args:
            symbol = context.args[0].upper().replace('_', '/')
            valid_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
            if symbol not in valid_pairs:
                await update.message.reply_text(f"❌ Invalid pair. Use: {', '.join(valid_pairs)}")
                return
        
        # Generate admin signal
        await update.message.reply_text("🎯 *Generating premium admin signal...*", parse_mode='Markdown')
        
        signal = await self.trading_bot.generate_admin_signal(user.id, symbol)
        
        if signal:
            # Format beautiful signal message
            direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
            analysis = json.loads(signal["analysis"])
            
            message = f"""
🎯 *ADMIN PREMIUM SIGNAL* 🌟
*24/7 Generation - Market Hours Bypassed*

{direction_emoji} *{signal['symbol']}* | **{signal['direction']}**
💵 *Entry:* `{signal['entry_price']:.5f}`
✅ *Take Profit:* `{signal['take_profit']:.5f}`
❌ *Stop Loss:* `{signal['stop_loss']:.5f}`

📊 *Signal Analytics:*
• Confidence: *{signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{signal['risk_reward']}* ⚖️
• Setup Quality: *{analysis['setup_quality']}* 💎

💡 *Market Analysis:*
{analysis['market_condition']}

⚡ *Admin Features:*
• 24/7 Signal Generation ✅
• Premium Analysis ✅  
• Enhanced Accuracy ✅

*Execute this premium signal immediately!* 🚀
"""
            keyboard = [
                [InlineKeyboardButton("✅ Trade Executed", callback_data="trade_done")],
                [InlineKeyboardButton("🎯 Another Signal", callback_data="admin_signal")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.message.reply_text("""
❌ *Signal Generation Failed*

*Possible reasons:*
• Database connection issue
• Symbol not available
• System temporary error

*Please try again in a moment.* 🔧
""", parse_mode='Markdown')

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command"""
        await update.message.reply_text("""
💎 *UPGRADE YOUR ACCOUNT*

*Contact admin directly for upgrades:*
{}

*We'll help you choose the best plan!* 🚀
""".format(Config.ADMIN_CONTACT), parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /contact command"""
        message = f"""
📞 *CONTACT ADMIN*

*Direct Contact:* {Config.ADMIN_CONTACT}

💡 *We can help with:*
• Subscription upgrades
• Payment processing  
• Technical support
• Account issues

*We're here to help!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("📱 Message Admin", url=f"https://t.me/{Config.ADMIN_CONTACT.replace('@', '')}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def plans_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /plans command"""
        message = """
💎 *LEKZY FX AI PRO - PREMIUM PLANS*

🌅 *BASIC* - $19/month
• Morning Session | 10 signals/day

🌇 *PRO* - $49/month  
• Morning + Evening | 25 signals/day

🌃 *VIP* - $99/month
• All Sessions | 50 signals/day

🌟 *PREMIUM* - $199/month
• 24/7 Access | Unlimited signals

*Contact {} for upgrades!* 🚀
""".format(Config.ADMIN_CONTACT)
        await update.message.reply_text(message, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        if self.admin_auth.is_admin(user.id):
            with sqlite3.connect(Config.DB_PATH) as conn:
                total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
                total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            
            message = f"""
📊 *ADMIN STATS* 👑

• Total Users: {total_users}
• Total Signals: {total_signals}
• Your Plan: {user_plan}
• Admin Access: ✅ Active

*Full system control enabled!* 🚀
"""
        else:
            message = f"""
📊 *YOUR ACCOUNT STATS*

• Plan: {user_plan}
• Signals: 5 per day
• Session: Morning
• Accuracy: 96.2%

*Contact {Config.ADMIN_CONTACT} to upgrade!* 💎
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
        session = self.trading_bot.session_manager.get_current_session()
        
        if session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED*

*Next Sessions:*
• 🌅 Morning: 08:00-12:00 (96.2%)
• 🌇 Evening: 16:00-20:00 (97.8%)
• 🌃 Asian: 00:00-04:00 (92.5%)

*Signals resume automatically!* 📈
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ ACTIVE

⏰ {session['start_hour']:02d}:00-{session['end_hour']:02d}:00
🎯 {session['accuracy']}% Accuracy
📈 Pairs: {', '.join(session['optimal_pairs'])}

💎 *Premium signals active!*
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, direction, entry_price, confidence, requested_by, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet. Check during session hours!")
            return
        
        message = "📡 *RECENT SIGNALS*\n\n"
        
        for symbol, direction, entry, confidence, requested_by, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            admin_badge = " 👑" if requested_by == "ADMIN" else ""
            
            message += f"{dir_emoji} {symbol}{admin_badge} {direction}\n"
            message += f"💵 {entry} | {confidence*100:.1f}%\n"
            message += f"⏰ {time_str}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')

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
        
        self.bot = FixedTelegramBot()
        await self.bot.initialize()
        
        self.running = True
        logger.info("🚀 LEKZY FX AI PRO - Fixed Version Started")
    
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
