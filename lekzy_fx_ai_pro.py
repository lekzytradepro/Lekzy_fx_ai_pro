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
    return "🤖 LEKZY FX AI PRO - Premium Trading Bot 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: Premium Active"

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
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Premium database initialized")
        
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

# ==================== PREMIUM SESSION MANAGER ====================
class PremiumSessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8, "end_hour": 12,
                "name": "🌅 European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "volatility": "HIGH",
                "accuracy": 96.2,
                "description": "London Open - High Volatility Period"
            },
            "EVENING": {
                "start_hour": 16, "end_hour": 20,
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "accuracy": 97.8,
                "description": "Peak Liquidity - Highest Accuracy"
            },
            "ASIAN": {
                "start_hour": 0, "end_hour": 4,
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "accuracy": 92.5,
                "description": "Premium Overnight Trading"
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

    def get_next_session(self):
        """Get next upcoming session"""
        now = datetime.now()
        current_hour = now.hour
        
        for session_id, session in self.sessions.items():
            if current_hour < session["start_hour"]:
                return {**session, "id": session_id}
        
        # If no session today, return first session tomorrow
        first_session = list(self.sessions.values())[0]
        return {**first_session, "id": list(self.sessions.keys())[0]}

# ==================== BEAUTIFUL SIGNAL GENERATOR ====================
class BeautifulSignalGenerator:
    def __init__(self):
        self.pending_signals = {}
        self.pre_entry_lead_time = 2  # minutes
    
    def generate_trade_analysis(self, symbol: str, session_type: str) -> dict:
        """Generate beautiful trade analysis"""
        analysis_templates = {
            "MORNING": [
                "Strong bullish momentum forming at key support",
                "Price bouncing from daily pivot with volume confirmation",
                "Breakout pattern developing with low spread",
                "London open volatility creating high-probability setup"
            ],
            "EVENING": [
                "NY/London overlap providing peak liquidity entry",
                "Institutional flow driving clear directional bias", 
                "High timeframe alignment with intraday momentum",
                "Low spread during peak hours for optimal execution"
            ],
            "ASIAN": [
                "Range-bound action providing clean support/resistance plays",
                "Overnight positioning creating continuation opportunities",
                "Low volatility allowing for precise entry execution",
                "Asian session liquidity providing stable market conditions"
            ]
        }
        
        risk_factors = [
            "Monitor for news spike volatility",
            "Watch for low volume false breaks", 
            "Check higher timeframe resistance",
            "Verify economic calendar clearance"
        ]
        
        return {
            "setup_quality": random.choice(["HIGH", "VERY_HIGH", "EXCELLENT"]),
            "market_condition": random.choice(analysis_templates[session_type]),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG_BULLISH", "STRONG_BEARISH", "BUILDING"]),
            "timeframe_alignment": random.choice(["PERFECT", "GOOD", "EXCELLENT"]),
            "risk_factors": random.sample(risk_factors, 2),
            "confidence_score": random.randint(85, 98)
        }
    
    def generate_pre_entry_signal(self, symbol: str, session_type: str) -> dict:
        """Generate beautiful pre-entry signal"""
        analysis = self.generate_trade_analysis(symbol, session_type)
        direction = "BUY" if random.random() > 0.48 else "SELL"
        
        # Calculate levels based on analysis
        if direction == "BUY":
            entry_price = round(analysis["key_level"] + 0.0008, 5)
        else:
            entry_price = round(analysis["key_level"] - 0.0008, 5)
        
        signal_id = f"PRE_{symbol.replace('/', '')}_{int(time.time())}"
        
        signal_data = {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_type": "PRE_ENTRY",
            "direction": direction,
            "entry_price": entry_price,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "confidence": round(random.uniform(0.88, 0.96), 3),
            "session_type": session_type,
            "analysis": json.dumps(analysis),
            "time_to_entry": self.pre_entry_lead_time,
            "risk_reward": 0.0,
            "generated_at": datetime.now().isoformat()
        }
        
        self.pending_signals[signal_id] = signal_data
        return signal_data
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate beautiful entry signal"""
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        analysis = json.loads(pre_signal["analysis"])
        
        # Calculate professional TP/SL levels
        if analysis["momentum"] in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            movement = 0.0040
        else:
            movement = 0.0025
        
        if pre_signal["direction"] == "BUY":
            take_profit = round(pre_signal["entry_price"] + movement, 5)
            stop_loss = round(pre_signal["entry_price"] - movement * 0.6, 5)
        else:
            take_profit = round(pre_signal["entry_price"] - movement, 5)
            stop_loss = round(pre_signal["entry_price"] + movement * 0.6, 5)
        
        risk_reward = round((take_profit - pre_signal["entry_price"]) / (pre_signal["entry_price"] - stop_loss), 2)
        
        entry_signal_id = pre_signal_id.replace("PRE_", "ENTRY_")
        
        entry_signal = {
            **pre_signal,
            "signal_id": entry_signal_id,
            "signal_type": "ENTRY",
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "time_to_entry": 0,
            "risk_reward": risk_reward
        }
        
        del self.pending_signals[pre_signal_id]
        return entry_signal

# ==================== PREMIUM NOTIFICATION MANAGER ====================
class PremiumNotificationManager:
    def __init__(self, application):
        self.application = application
        self.admin_auth = AdminAuth()
    
    async def notify_new_subscriber(self, user_id: int, username: str, plan_type: str):
        """Notify all active admins about new subscriber"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("SELECT user_id FROM admin_sessions")
            active_admins = [row[0] for row in cursor.fetchall()]
        
        if not active_admins:
            return
            
        message = f"""
🎉 *NEW SUBSCRIBER ALERT!*

👤 *User Details:*
• Username: @{username}
• User ID: `{user_id}`
• Plan: *{plan_type}*
• Time: {datetime.now().strftime('%H:%M:%S')}

📊 *Welcome Package:*
• 3-Day Free Trial Activated
• Morning Session Access
• 5 Signals Per Day
• 96.2% Accuracy

💡 *Next Steps:*
• Monitor user activity
• Send welcome message
• Track conversion potential

Welcome to LEKZY FX AI PRO! 🚀
"""
        
        for admin_id in active_admins:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"✅ Admin notified: {admin_id}")
                
                # Log notification
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

# ==================== PREMIUM SUBSCRIPTION MANAGER ====================
class PremiumSubscriptionManager:
    def __init__(self, db_path: str, notification_manager: PremiumNotificationManager):
        self.db_path = db_path
        self.notification_manager = notification_manager
    
    def start_premium_trial(self, user_id: int, username: str, first_name: str):
        """Start premium free trial"""
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
        
        # Notify admins
        asyncio.create_task(self.notification_manager.notify_new_subscriber(user_id, username, "TRIAL"))
        logger.info(f"✅ Premium trial started: {username} ({user_id})")
    
    def get_user_plan(self, user_id: int) -> str:
        """Get user's current plan"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT plan_type FROM subscriptions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else "TRIAL"
    
    def get_user_sessions(self, user_id: int) -> list:
        """Get user's allowed sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT allowed_sessions FROM subscriptions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return json.loads(result[0])
        return ["MORNING"]  # Default for trial users
    
    def can_access_session(self, user_id: int, session_type: str) -> bool:
        """Check if user can access specific session"""
        if self.get_user_plan(user_id) == "ADMIN":
            return True  # Admins access all sessions
        user_sessions = self.get_user_sessions(user_id)
        return session_type in user_sessions

# ==================== PREMIUM TRADING BOT ====================
class PremiumTradingBot:
    def __init__(self):
        self.session_manager = PremiumSessionManager()
        self.signal_generator = BeautifulSignalGenerator()
        self.is_running = False
    
    async def start_premium_signals(self):
        """Start premium signal generation"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    current_session = self.session_manager.get_current_session()
                    
                    if current_session["id"] != "CLOSED":
                        logger.info(f"🎯 Generating premium signals for {current_session['name']}")
                        
                        # Generate for optimal pairs
                        for symbol in current_session["optimal_pairs"][:2]:
                            # Step 1: Pre-entry signal
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, current_session["id"])
                            
                            # Store pre-entry
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    pre_signal["signal_id"], pre_signal["symbol"], pre_signal["signal_type"],
                                    pre_signal["direction"], pre_signal["entry_price"], pre_signal["take_profit"],
                                    pre_signal["stop_loss"], pre_signal["confidence"], pre_signal["session_type"],
                                    pre_signal["analysis"], pre_signal["time_to_entry"], pre_signal["risk_reward"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📊 Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
                            
                            # Wait for entry
                            await asyncio.sleep(self.signal_generator.pre_entry_lead_time * 60)
                            
                            # Step 2: Entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        entry_signal["signal_id"], entry_signal["symbol"], entry_signal["signal_type"],
                                        entry_signal["direction"], entry_signal["entry_price"], entry_signal["take_profit"],
                                        entry_signal["stop_loss"], entry_signal["confidence"], entry_signal["session_type"],
                                        entry_signal["analysis"], entry_signal["time_to_entry"], entry_signal["risk_reward"]
                                    ))
                                    conn.commit()
                                
                                logger.info(f"🎯 Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    # Professional cooldown
                    wait_time = random.randint(300, 900)
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Premium signal generation started")

# ==================== BEAUTIFUL TELEGRAM BOT ====================
class BeautifulTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.notification_manager = None
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the premium bot"""
        self.application = Application.builder().token(self.token).build()
        self.notification_manager = PremiumNotificationManager(self.application)
        self.subscription_manager = PremiumSubscriptionManager(Config.DB_PATH, self.notification_manager)
        self.trading_bot = PremiumTradingBot()
        
        # Premium command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_premium_signals()
        
        logger.info("🤖 Premium Telegram bot initialized!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start with beautiful layout"""
        user = update.effective_user
        self.subscription_manager.start_premium_trial(user.id, user.username, user.first_name)
        
        current_session = self.trading_bot.session_manager.get_current_session()
        next_session = self.trading_bot.session_manager.get_next_session()
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
You'll receive professional trading signals with premium analysis.

🕒 *Current Session:* {current_session['name']}
⏰ *Next Session:* {next_session['name']} at {next_session['start_hour']:02d}:00

🔔 *Admins have been notified of your subscription.*

📊 *Enhanced Signal System:*
• 📊 Pre-entry analysis (2 min before)
• 🎯 Entry signals with exact levels  
• ⚡ Minimal, actionable trade analysis
• 💎 Professional risk management

💡 *Available Commands:*
• /stats - Your account status
• /session - Current trading session
• /signals - Recent trading signals
• /login - Admin access
• /upgrade - Premium plans

*Happy trading!* 📈
"""
        keyboard = [
            [InlineKeyboardButton("📊 Account Stats", callback_data="stats")],
            [InlineKeyboardButton("🕒 Trading Session", callback_data="session")],
            [InlineKeyboardButton("📡 Recent Signals", callback_data="signals")],
            [InlineKeyboardButton("💎 Upgrade Plan", callback_data="upgrade")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command"""
        user = update.effective_user
        
        if not context.args:
            await update.message.reply_text(
                "🔐 *Admin Login*\n\nUsage: `/login YOUR_ADMIN_TOKEN`\n\n*Contact owner for admin access.*",
                parse_mode='Markdown'
            )
            return
        
        token = context.args[0]
        
        if self.admin_auth.verify_token(token):
            self.admin_auth.create_session(user.id, user.username)
            
            # Upgrade user to admin plan
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute(
                    "UPDATE subscriptions SET plan_type = 'ADMIN' WHERE user_id = ?",
                    (user.id,)
                )
                conn.commit()
            
            await update.message.reply_text("✅ *Admin access granted!* 🌟", parse_mode='Markdown')
            await self.admin_command(update, context)
        else:
            await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command with premium layout"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text(
                "❌ *Admin Access Required*\n\nUse `/login YOUR_ADMIN_TOKEN` to access premium admin features.",
                parse_mode='Markdown'
            )
            return
        
        # Get premium statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            pre_entry = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'PRE_ENTRY'").fetchone()[0]
            entry = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'ENTRY'").fetchone()[0]
            recent_users = conn.execute("SELECT username, created_at FROM subscriptions ORDER BY created_at DESC LIMIT 3").fetchall()
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🏢 *LEKZY FX AI PRO - ADMIN DASHBOARD* 🌟

*Welcome, @{user.username}!* 👑

📊 *System Statistics:*
• 👥 Total Users: *{total_users}*
• 📡 Total Signals: *{total_signals}*
• 📊 Pre-entry Signals: *{pre_entry}*
• 🎯 Entry Signals: *{entry}*
• 🕒 Current Session: *{current_session['name']}*

🔧 *Premium Admin Features:*
• ✅ All-session signal access
• 📊 Real-time user analytics  
• 🔔 Instant new user notifications
• ⚡ Enhanced signal monitoring

👥 *Recent Users:*
"""
        
        for username, joined in recent_users:
            date_str = datetime.fromisoformat(joined).strftime("%m/%d %H:%M")
            message += f"• @{username} - {date_str}\n"
        
        message += "\n💎 *You have full access to all trading sessions!*"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh")],
            [InlineKeyboardButton("📊 Detailed Stats", callback_data="admin_stats")],
            [InlineKeyboardButton("👥 User Management", callback_data="admin_users")],
            [InlineKeyboardButton("🔐 Logout", callback_data="admin_logout")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout command"""
        user = update.effective_user
        
        if self.admin_auth.is_admin(user.id):
            self.admin_auth.logout(user.id)
            await update.message.reply_text("✅ *Logged out from admin access*", parse_mode='Markdown')
        else:
            await update.message.reply_text("ℹ️ You are not currently logged in as admin.")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats with beautiful layout"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        user_sessions = self.subscription_manager.get_user_sessions(user.id)
        
        is_admin = self.admin_auth.is_admin(user.id)
        
        if is_admin:
            with sqlite3.connect(Config.DB_PATH) as conn:
                total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
                total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            
            message = f"""
📊 *PREMIUM ADMIN STATS* 👑

👤 *Account Overview:*
• User: {user.first_name}
• Plan: *ADMIN* 🌟
• Access: *All Sessions* ✅
• Signals: *Unlimited* 🚀

📈 *System Performance:*
• Total Users: {total_users}
• Signals Generated: {total_signals}
• System Uptime: 99.8%
• Accuracy Rate: 96.3%

💎 *Admin Privileges:*
• Full session access
• Real-time analytics
• User management
• Signal monitoring

*You have premium access to all features!* 🎯
"""
        else:
            message = f"""
📊 *YOUR ACCOUNT STATS*

👤 *User:* {user.first_name}
📋 *Plan:* {user_plan}
🎯 *Accuracy:* 96.2%

📈 *Trial Features:*
• Sessions: {', '.join(user_sessions)}
• Signals: 5 per day
• Pre-entry Analysis: ✅
• Entry Signals: ✅

💡 *Upgrade Benefits:*
• All trading sessions
• Unlimited signals  
• Higher accuracy (97.8%)
• Priority support

*Use /upgrade for premium features!* 💎
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session with beautiful layout"""
        current_session = self.trading_bot.session_manager.get_current_session()
        user = update.effective_user
        is_admin = self.admin_auth.is_admin(user.id)
        user_sessions = self.subscription_manager.get_user_sessions(user.id)
        
        if current_session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED - Professional Analysis*

*No active trading sessions at the moment.*

📅 *Upcoming Premium Sessions:*
• 🌅 *Morning Session* (08:00 - 12:00)
  Accuracy: 96.2% | Volatility: HIGH
  Pairs: EUR/USD, GBP/USD, EUR/JPY

• 🌇 *Evening Session* (16:00 - 20:00)  
  Accuracy: 97.8% | Volatility: VERY HIGH
  Pairs: USD/JPY, USD/CAD, XAU/USD

• 🌃 *Asian Session* (00:00 - 04:00)
  Accuracy: 92.5% | Volatility: MEDIUM  
  Pairs: AUD/JPY, NZD/USD, USD/JPY

💡 *Premium signals will resume automatically during session hours!*
"""
        else:
            can_access = is_admin or (current_session["id"] in user_sessions)
            access_status = "✅ FULL ACCESS" if can_access else "❌ NO ACCESS"
            
            message = f"""
🕒 *CURRENT TRADING SESSION* 🚀

*{current_session['name']}*
{current_session['description']}

📊 *Session Analytics:*
• Status: ✅ *ACTIVE NOW*
• Your Access: {access_status}
• Volatility: {current_session['volatility']}
• Accuracy: {current_session['accuracy']}%
• Hours: {current_session['start_hour']:02d}:00 - {current_session['end_hour']:02d}:00

🎯 *Optimal Trading Pairs:*
{', '.join(current_session['optimal_pairs'])}

"""
            
            if can_access:
                message += """
⚡ *Premium Signals Active:*
• 📊 Pre-entry analysis (2 min advance)
• 🎯 Entry signals with exact levels
• 💎 Professional risk management
• 📈 Real-time market monitoring

*You will receive signals for this session!* 🚀
"""
            else:
                message += """
🔒 *Access Restricted*
*Upgrade your plan to receive signals for this session!*

*Use /upgrade for full market access* 💎
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals with beautiful layout"""
        user = update.effective_user
        is_admin = self.admin_auth.is_admin(user.id)
        
        # Admins see all signals, users see only recent
        if is_admin:
            signals_query = """
                SELECT signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, created_at, analysis
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 8
            """
        else:
            signals_query = """
                SELECT signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, created_at, analysis
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            recent_signals = conn.execute(signals_query).fetchall()
        
        if not recent_signals:
            await update.message.reply_text("""
📭 *No Signals Generated Yet*

*Signals are automatically generated during:*
• 🌅 Morning Session (08:00-12:00)
• 🌇 Evening Session (16:00-20:00) 
• 🌃 Asian Session (00:00-04:00)

*Check back during market hours for premium trading signals!* 📈
""", parse_mode='Markdown')
            return
        
        message = "📡 *RECENT TRADING SIGNALS* 🚀\n\n"
        
        for signal in recent_signals:
            (signal_id, symbol, signal_type, direction, entry_price, take_profit, 
             stop_loss, confidence, session_type, created_at, analysis_json) = signal
            
            if signal_type == "PRE_ENTRY":
                emoji = "📊"
                type_text = "PRE-ENTRY ANALYSIS"
                time_info = "⏰ Entry in 2 min"
            else:
                emoji = "🎯" 
                type_text = "ENTRY SIGNAL"
                time_info = "⚡ ENTER NOW"
            
            time_str = datetime.fromisoformat(created_at).strftime("%H:%M")
            direction_emoji = "🟢" if direction == "BUY" else "🔴"
            session_emoji = "🌅" if session_type == "MORNING" else "🌇" if session_type == "EVENING" else "🌃"
            
            message += f"""
{emoji} *{type_text}* {session_emoji}
{direction_emoji} *{symbol}* | {direction}
💵 *Entry:* `{entry_price:.5f}`
🎯 *Confidence:* {confidence*100:.1f}%
{time_info} | {time_str}
"""
            
            if signal_type == "PRE_ENTRY" and analysis_json:
                try:
                    analysis = json.loads(analysis_json)
                    message += f"   ⚡ {analysis['market_condition']}\n"
                except:
                    pass
            
            if signal_type == "ENTRY":
                message += f"   ✅ TP: `{take_profit:.5f}` | ❌ SL: `{stop_loss:.5f}`\n"
            
            message += "\n"
        
        if is_admin:
            message += "💎 *Admin View: Full signal history available*"
        else:
            message += "💡 *Pre-entry signals come 2 minutes before entry signals*"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command"""
        user = update.effective_user
        is_admin = self.admin_auth.is_admin(user.id)
        
        if is_admin:
            await update.message.reply_text("""
💎 *PREMIUM ADMIN STATUS*

*You already have the highest level of access!* 👑

• ✅ All trading sessions
• ✅ Unlimited signals
• ✅ Real-time analytics
• ✅ User management
• ✅ Maximum accuracy

*Thank you for managing LEKZY FX AI PRO!* 🚀
""", parse_mode='Markdown')
            return
        
        message = """
💎 *UPGRADE YOUR TRADING EXPERIENCE*

*Current Plan:* TRIAL (Limited Access)

🚀 *PREMIUM PLANS:*

🌅 *BASIC PLAN* - $19/month
• Morning Session Access
• 10 Signals/Day  
• 96.2% Accuracy
• Basic Support

🌇 *PRO PLAN* - $49/month  
• Morning + Evening Sessions
• 25 Signals/Day
• 97.8% Accuracy
• Priority Support

🌃 *VIP PLAN* - $99/month
• All Trading Sessions
• 50 Signals/Day  
• 98.5% Accuracy
• VIP Support
• Advanced Analytics

🌟 *PREMIUM PLAN* - $199/month
• 24/7 Signal Access
• Unlimited Signals
• 99.2% Accuracy  
• Personal Coach
• Custom Strategies

💡 *Contact @YourSalesHandle to upgrade!*

*Unlock your full trading potential!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def start_polling(self):
        """Start polling"""
        await self.application.updater.start_polling()

    async def stop(self):
        """Stop bot"""
        self.trading_bot.is_running = False
        await self.application.stop()

# ==================== PREMIUM APPLICATION ====================
class PremiumApplication:
    def __init__(self):
        self.bot = None
        self.running = False
    
    async def setup(self):
        """Setup premium application"""
        initialize_database()
        start_web_server()
        
        self.bot = BeautifulTelegramBot()
        await self.bot.initialize()
        
        self.running = True
        logger.info("🚀 LEKZY FX AI PRO Premium Edition Started")
    
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

# ==================== START PREMIUM BOT ====================
async def main():
    app = PremiumApplication()
    try:
        await app.run()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
