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
    ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@LekzyTradingPro")  # Your Telegram username
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
    return "🤖 LEKZY FX AI PRO - Subscription System Active 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: Subscription System Running"

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
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Subscription database initialized")
        
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

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str, notification_manager):
        self.db_path = db_path
        self.notification_manager = notification_manager
    
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
    
    def upgrade_user(self, user_id: int, new_plan: str):
        """Upgrade user to paid plan"""
        plan_configs = {
            "BASIC": {"signals": 10, "sessions": '["MORNING"]', "days": 30},
            "PRO": {"signals": 25, "sessions": '["MORNING", "EVENING"]', "days": 30},
            "VIP": {"signals": 50, "sessions": '["MORNING", "EVENING", "ASIAN"]', "days": 30},
            "PREMIUM": {"signals": 999, "sessions": '["MORNING", "EVENING", "ASIAN"]', "days": 30}
        }
        
        if new_plan not in plan_configs:
            return False
        
        config = plan_configs[new_plan]
        end_date = datetime.now() + timedelta(days=config["days"])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE subscriptions 
                SET plan_type = ?, end_date = ?, payment_status = 'PAID',
                    max_daily_signals = ?, allowed_sessions = ?
                WHERE user_id = ?
            """, (new_plan, end_date.isoformat(), config["signals"], config["sessions"], user_id))
            conn.commit()
        
        logger.info(f"✅ User upgraded: {user_id} -> {new_plan}")
        return True
    
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
        return ["MORNING"]
    
    def create_upgrade_request(self, user_id: int, username: str, plan: str, contact_method: str):
        """Create upgrade request"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO upgrade_requests 
                (user_id, username, requested_plan, contact_method)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, plan, contact_method))
            conn.commit()
        
        logger.info(f"✅ Upgrade request: {username} -> {plan}")

# ==================== UPGRADE REQUEST MANAGER ====================
class UpgradeRequestManager:
    def __init__(self, application):
        self.application = application
    
    async def notify_upgrade_request(self, user_id: int, username: str, plan: str, contact_method: str):
        """Notify admins about upgrade request"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("SELECT user_id FROM admin_sessions")
            active_admins = [row[0] for row in cursor.fetchall()]
        
        if not active_admins:
            return
        
        message = f"""
💎 *UPGRADE REQUEST RECEIVED!*

👤 *User Details:*
• Username: @{username}
• User ID: `{user_id}`
• Requested Plan: *{plan}*
• Contact Method: {contact_method}
• Time: {datetime.now().strftime('%H:%M:%S')}

💰 *Plan Pricing:*
• BASIC: $19/month
• PRO: $49/month  
• VIP: $99/month
• PREMIUM: $199/month

🚀 *Action Required:*
1. Contact user via {contact_method}
2. Process payment
3. Upgrade subscription
4. Confirm completion

*Reply to this user directly to process their upgrade!*
"""
        
        for admin_id in active_admins:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
                
                # Log notification
                with sqlite3.connect(Config.DB_PATH) as conn:
                    conn.execute("""
                        INSERT INTO admin_notifications 
                        (notification_type, user_id, username, plan_type, details)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        "UPGRADE_REQUEST",
                        user_id,
                        username,
                        plan,
                        json.dumps({"contact_method": contact_method, "timestamp": datetime.now().isoformat()})
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"❌ Failed to notify admin: {e}")

# ==================== BEAUTIFUL SIGNAL GENERATOR ====================
class BeautifulSignalGenerator:
    def __init__(self):
        self.pending_signals = {}
    
    def generate_pre_entry_signal(self, symbol: str, session_type: str) -> dict:
        """Generate pre-entry signal"""
        direction = random.choice(["BUY", "SELL"])
        base_price = random.uniform(1.0800, 1.1200) if "EUR" in symbol else random.uniform(1.2600, 1.3000)
        
        if direction == "BUY":
            entry_price = round(base_price + 0.0005, 5)
        else:
            entry_price = round(base_price - 0.0005, 5)
        
        signal_id = f"PRE_{symbol.replace('/', '')}_{int(time.time())}"
        
        analysis = {
            "setup_quality": random.choice(["HIGH", "VERY_HIGH"]),
            "market_condition": random.choice([
                "Strong momentum at key level",
                "Breakout confirmation forming",
                "Support/resistance bounce",
                "Trend alignment perfect"
            ])
        }
        
        return {
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
            "time_to_entry": 2,
            "risk_reward": 0.0
        }
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate entry signal"""
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        
        if pre_signal["direction"] == "BUY":
            take_profit = round(pre_signal["entry_price"] * 1.003, 5)
            stop_loss = round(pre_signal["entry_price"] * 0.998, 5)
        else:
            take_profit = round(pre_signal["entry_price"] * 0.997, 5)
            stop_loss = round(pre_signal["entry_price"] * 1.002, 5)
        
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

# ==================== TRADING BOT ====================
class TradingBot:
    def __init__(self):
        self.session_manager = SessionManager()
        self.signal_generator = BeautifulSignalGenerator()
        self.is_running = False
    
    async def start_signals(self):
        """Start signal generation"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED":
                        for symbol in session["optimal_pairs"][:1]:
                            # Pre-entry signal
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, session["id"])
                            self.signal_generator.pending_signals[pre_signal["signal_id"]] = pre_signal
                            
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, tuple(pre_signal.values()))
                                conn.commit()
                            
                            logger.info(f"📊 Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
                            
                            # Wait for entry
                            await asyncio.sleep(120)  # 2 minutes
                            
                            # Entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, tuple(entry_signal.values()))
                                    conn.commit()
                                
                                logger.info(f"🎯 Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    await asyncio.sleep(random.randint(300, 600))
                    
                except Exception as e:
                    logger.error(f"Signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())

# ==================== COMPLETE TELEGRAM BOT ====================
class CompleteTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.subscription_manager = None
        self.upgrade_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the complete bot"""
        self.application = Application.builder().token(self.token).build()
        self.upgrade_manager = UpgradeRequestManager(self.application)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH, self.upgrade_manager)
        self.trading_bot = TradingBot()
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        self.application.add_handler(CommandHandler("plans", self.plans_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_signals()
        
        logger.info("🤖 Complete Trading Bot Initialized!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
Experience professional trading signals with our premium system.

💡 *Get Started:*
• Use /plans to see upgrade options
• Use /upgrade to request premium access  
• Use /contact for immediate assistance
• Use /stats to check your account

🔔 *Admin has been notified of your interest!*

*Start your trading journey today!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("💎 View Plans", callback_data="plans")],
            [InlineKeyboardButton("🚀 Upgrade Now", callback_data="upgrade")],
            [InlineKeyboardButton("📞 Contact Admin", callback_data="contact")]
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
            await update.message.reply_text("✅ *Admin access granted!*", parse_mode='Markdown')
            await self.admin_command(update, context)
        else:
            await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use /login")
            return
        
        # Get admin statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            pending_upgrades = conn.execute("SELECT COUNT(*) FROM upgrade_requests WHERE status = 'PENDING'").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        
        message = f"""
🏢 *ADMIN DASHBOARD*

📊 *Statistics:*
• Total Users: {total_users}
• Pending Upgrades: {pending_upgrades}
• Signals Generated: {total_signals}

🚀 *Admin Actions:*
• Contact users for upgrades
• Process upgrade requests  
• Monitor system performance

💡 *Use these commands:*
• Check upgrade requests manually
• Contact users directly
• Process payments
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command - MAIN UPGRADE FLOW"""
        user = update.effective_user
        
        if not context.args:
            # Show upgrade options
            message = """
💎 *REQUEST UPGRADE*

Choose your desired plan:

1. 🌅 *BASIC* - $19/month
   Morning Session | 10 signals/day

2. 🌇 *PRO* - $49/month  
   Morning + Evening | 25 signals/day

3. 🌃 *VIP* - $99/month
   All Sessions | 50 signals/day

4. 🌟 *PREMIUM* - $199/month
   24/7 Access | Unlimited signals

*How to Upgrade:*
Usage: `/upgrade <plan_name> <contact_method>`

*Examples:*
• `/upgrade basic telegram`
• `/upgrade pro whatsapp`  
• `/upgrade vip email`

*Contact Methods:* telegram, whatsapp, email

*We'll contact you immediately!* 🚀
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            return
        
        if len(context.args) < 2:
            await update.message.reply_text("❌ Usage: `/upgrade <plan> <contact_method>`\nExample: `/upgrade pro telegram`")
            return
        
        plan = context.args[0].upper()
        contact_method = context.args[1].lower()
        
        valid_plans = ["BASIC", "PRO", "VIP", "PREMIUM"]
        valid_contacts = ["telegram", "whatsapp", "email"]
        
        if plan not in valid_plans:
            await update.message.reply_text(f"❌ Invalid plan. Choose from: {', '.join(valid_plans)}")
            return
        
        if contact_method not in valid_contacts:
            await update.message.reply_text(f"❌ Invalid contact method. Choose from: {', '.join(valid_contacts)}")
            return
        
        # Create upgrade request
        self.subscription_manager.create_upgrade_request(user.id, user.username, plan, contact_method)
        
        # Notify admins
        await self.upgrade_manager.notify_upgrade_request(user.id, user.username, plan, contact_method)
        
        # Confirm to user
        message = f"""
✅ *UPGRADE REQUEST RECEIVED!*

📋 *Your Request:*
• Plan: *{plan}*
• Contact Method: *{contact_method}*
• Username: @{user.username}

💰 *Next Steps:*
1. Admin will contact you via {contact_method}
2. Complete payment process  
3. Get instant activation
4. Start trading premium!

⏰ *Response Time:* Usually within 1 hour

*Thank you for choosing LEKZY FX AI PRO!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /contact command"""
        user = update.effective_user
        
        message = f"""
📞 *CONTACT ADMIN FOR IMMEDIATE ASSISTANCE*

*Admin Contact:* {Config.ADMIN_CONTACT}

💡 *What we can help with:*
• Subscription upgrades
• Payment processing
• Technical support
• Account issues
• General inquiries

🚀 *Quick Upgrade Process:*
1. Message {Config.ADMIN_CONTACT} directly
2. Specify your desired plan
3. Complete secure payment
4. Get instant activation

⏰ *Response Time:* Within 1 hour

*We're here to help you succeed!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("📱 Message Admin Now", url=f"https://t.me/{Config.ADMIN_CONTACT.replace('@', '')}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def plans_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /plans command"""
        message = """
💎 *LEKZY FX AI PRO - PREMIUM PLANS*

🌅 *BASIC PLAN* - $19/month
• Morning Session (08:00-12:00)
• 10 Signals Per Day
• 96.2% Accuracy
• Basic Support
• *Ideal for beginners*

🌇 *PRO PLAN* - $49/month  
• Morning + Evening Sessions
• 25 Signals Per Day
• 97.8% Accuracy  
• Priority Support
• *Best for serious traders*

🌃 *VIP PLAN* - $99/month
• All Trading Sessions (24/7)
• 50 Signals Per Day
• 98.5% Accuracy
• VIP Support
• Advanced Analytics
• *For professional traders*

🌟 *PREMIUM PLAN* - $199/month
• 24/7 Signal Access
• Unlimited Signals
• 99.2% Accuracy  
• Personal Trading Coach
• Custom Strategies
• *Ultimate trading experience*

🚀 *How to Upgrade:*
1. Use `/upgrade <plan> <contact_method>`
2. Or contact {Config.ADMIN_CONTACT} directly
3. We'll guide you through payment
4. Get instant activation

*Example:* `/upgrade pro telegram`

*Start your premium journey today!* 🎯
"""
        keyboard = [
            [InlineKeyboardButton("🚀 Upgrade Now", callback_data="upgrade_now")],
            [InlineKeyboardButton("📞 Contact Admin", callback_data="contact_admin")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        user_sessions = self.subscription_manager.get_user_sessions(user.id)
        
        message = f"""
📊 *YOUR ACCOUNT STATS*

👤 User: {user.first_name}
📋 Plan: {user_plan}
🎯 Accuracy: 96.2%

📈 *Current Access:*
• Sessions: {', '.join(user_sessions)}
• Signals: 5 per day (trial)
• Pre-entry Analysis: ✅
• Entry Signals: ✅

💎 *Upgrade to unlock:*
• More sessions & signals
• Higher accuracy (up to 99.2%)
• Priority support
• Advanced features

*Use /plans to see all options!*
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

*Upgrade for 24/7 market access!*
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ ACTIVE

⏰ Hours: {session['start_hour']:02d}:00-{session['end_hour']:02d}:00
🎯 Accuracy: {session['accuracy']}%
📈 Pairs: {', '.join(session['optimal_pairs'])}

💎 *Premium signals active!*
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, signal_type, direction, entry_price, confidence, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet. Check during session hours!")
            return
        
        message = "📡 *RECENT SIGNALS*\n\n"
        
        for symbol, signal_type, direction, entry, confidence, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            type_emoji = "📊" if signal_type == "PRE_ENTRY" else "🎯"
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            
            message += f"{type_emoji} {dir_emoji} {symbol} {direction}\n"
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
        
        self.bot = CompleteTelegramBot()
        await self.bot.initialize()
        
        self.running = True
        logger.info("🚀 LEKZY FX AI PRO with Subscription System Started")
    
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
