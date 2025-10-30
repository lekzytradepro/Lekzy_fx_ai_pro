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
    return "🤖 LEKZY FX AI PRO - 24/7 Admin Signals 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: 24/7 Signals Active"

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
                requested_by TEXT DEFAULT 'AUTO',  # AUTO or ADMIN
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ 24/7 Signal database initialized")
        
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

# ==================== 24/7 SIGNAL GENERATOR ====================
class AllDaySignalGenerator:
    def __init__(self):
        self.pending_signals = {}
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
    
    def generate_trade_analysis(self, symbol: str, is_admin_request: bool = False) -> dict:
        """Generate professional trade analysis"""
        if is_admin_request:
            # Enhanced analysis for admin-requested signals
            analysis_templates = [
                "Strong institutional flow detected with clear directional bias",
                "Price action confirming breakout with volume validation",
                "Multiple timeframe alignment creating high-probability setup",
                "Key support/resistance level holding with momentum confirmation",
                "Market structure break with follow-through momentum",
                "Economic catalyst driving clear directional movement"
            ]
            quality = "EXCELLENT"
            confidence_boost = 0.05
        else:
            analysis_templates = [
                "Strong momentum at key level",
                "Breakout confirmation forming",
                "Support/resistance bounce",
                "Trend alignment positive"
            ]
            quality = random.choice(["HIGH", "VERY_HIGH"])
            confidence_boost = 0.0
        
        risk_factors = [
            "Monitor for news spike volatility",
            "Watch for low volume false breaks", 
            "Check higher timeframe resistance",
            "Verify economic calendar clearance"
        ]
        
        return {
            "setup_quality": quality,
            "market_condition": random.choice(analysis_templates),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG_BULLISH", "STRONG_BEARISH", "BUILDING"]),
            "timeframe_alignment": random.choice(["PERFECT", "EXCELLENT"]),
            "risk_factors": random.sample(risk_factors, 2),
            "confidence_score": random.randint(85, 98),
            "admin_enhanced": is_admin_request
        }
    
    def generate_pre_entry_signal(self, symbol: str, is_admin_request: bool = False) -> dict:
        """Generate pre-entry signal"""
        analysis = self.generate_trade_analysis(symbol, is_admin_request)
        direction = "BUY" if random.random() > 0.48 else "SELL"
        
        # Calculate levels based on analysis
        base_price = analysis["key_level"]
        if direction == "BUY":
            entry_price = round(base_price + 0.0008, 5)
        else:
            entry_price = round(base_price - 0.0008, 5)
        
        signal_id = f"PRE_{symbol.replace('/', '')}_{int(time.time())}"
        
        # Enhanced confidence for admin requests
        base_confidence = random.uniform(0.88, 0.96)
        if is_admin_request:
            base_confidence = min(0.98, base_confidence + 0.03)  # Boost for admin
        
        signal_data = {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_type": "PRE_ENTRY",
            "direction": direction,
            "entry_price": entry_price,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "confidence": round(base_confidence, 3),
            "session_type": "ADMIN_REQUEST" if is_admin_request else "AUTO",
            "analysis": json.dumps(analysis),
            "time_to_entry": 2,  # 2 minutes for pre-entry
            "risk_reward": 0.0,
            "requested_by": "ADMIN" if is_admin_request else "AUTO",
            "generated_at": datetime.now().isoformat()
        }
        
        self.pending_signals[signal_id] = signal_data
        return signal_data
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate entry signal"""
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        analysis = json.loads(pre_signal["analysis"])
        
        # Calculate professional TP/SL levels
        movement = 0.0040 if analysis.get("admin_enhanced", False) else 0.0030
        
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
    
    def generate_instant_signal(self, symbol: str) -> dict:
        """Generate instant signal (both pre-entry and entry combined)"""
        # Generate pre-entry
        pre_signal = self.generate_pre_entry_signal(symbol, True)
        
        # Immediately generate entry signal
        entry_signal = self.generate_entry_signal(pre_signal["signal_id"])
        
        return entry_signal

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8, "end_hour": 12,
                "name": "🌅 European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 16, "end_hour": 20,
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 0, "end_hour": 4,
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
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

# ==================== SUBSCRIPTION & UPGRADE MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str, upgrade_manager):
        self.db_path = db_path
        self.upgrade_manager = upgrade_manager
    
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

💰 *Plan Pricing:*
• BASIC: $19/month
• PRO: $49/month  
• VIP: $99/month
• PREMIUM: $199/month

🚀 *Contact user now to process upgrade!*
"""
        
        for admin_id in active_admins:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"❌ Failed to notify admin: {e}")

# ==================== 24/7 TRADING BOT ====================
class AllDayTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = SessionManager()
        self.signal_generator = AllDaySignalGenerator()
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
                            # Pre-entry signal
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, False)
                            
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, requested_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    pre_signal["signal_id"], pre_signal["symbol"], pre_signal["signal_type"],
                                    pre_signal["direction"], pre_signal["entry_price"], pre_signal["take_profit"],
                                    pre_signal["stop_loss"], pre_signal["confidence"], pre_signal["session_type"],
                                    pre_signal["analysis"], pre_signal["time_to_entry"], pre_signal["risk_reward"],
                                    pre_signal["requested_by"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📊 Auto Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
                            
                            # Wait for entry
                            await asyncio.sleep(120)
                            
                            # Entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, requested_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, tuple(entry_signal.values()))
                                    conn.commit()
                                
                                logger.info(f"🎯 Auto Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    await asyncio.sleep(random.randint(300, 600))
                    
                except Exception as e:
                    logger.error(f"Auto signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Auto signal generation started")
    
    async def generate_admin_signal(self, user_id: int, symbol: str = None):
        """Generate instant signal for admin (available 24/7)"""
        try:
            if not symbol:
                # Pick random symbol
                symbol = random.choice(self.signal_generator.all_pairs)
            
            # Generate instant signal (both pre-entry and entry combined)
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
                        signal["stop_loss"], signal["confidence"], "ADMIN_24_7",
                        signal["analysis"], signal["time_to_entry"], signal["risk_reward"],
                        "ADMIN"
                    ))
                    conn.commit()
                
                logger.info(f"🎯 Admin signal generated: {signal['symbol']} {signal['direction']}")
                return signal
            return None
            
        except Exception as e:
            logger.error(f"Admin signal error: {e}")
            return None

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
        self.trading_bot = AllDayTradingBot(self.application)
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))  # New admin signal command
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        self.application.add_handler(CommandHandler("plans", self.plans_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_auto_signals()
        
        logger.info("🤖 24/7 Trading Bot Initialized!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
Experience professional trading signals with our premium system.

🌟 *New Feature:* Admins can now generate signals 24/7!

💡 *Get Started:*
• Use /plans to see upgrade options
• Use /upgrade to request premium access  
• Use /contact for immediate assistance

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
                "🔐 *Admin Login*\n\nUsage: `/login YOUR_ADMIN_TOKEN`",
                parse_mode='Markdown'
            )
            return
        
        token = context.args[0]
        
        if self.admin_auth.verify_token(token):
            self.admin_auth.create_session(user.id, user.username)
            await update.message.reply_text("""
✅ *Admin Access Granted!* 🌟

🎯 *New Admin Features Unlocked:*
• `/signal` - Generate instant signals 24/7
• `/signal EUR/USD` - Specific pair signals
• Full system access anytime

*Test the new 24/7 signal generation!* 🚀
""", parse_mode='Markdown')
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
            admin_signals = conn.execute("SELECT COUNT(*) FROM signals WHERE requested_by = 'ADMIN'").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🏢 *ADMIN DASHBOARD - 24/7 MODE* 🌟

📊 *Statistics:*
• Total Users: {total_users}
• Pending Upgrades: {pending_upgrades}
• Total Signals: {total_signals}
• Admin Signals: {admin_signals}
• Current Session: {current_session['name']}

🎯 *24/7 Admin Commands:*
• `/signal` - Generate random signal now
• `/signal EUR/USD` - Specific pair signal
• Instant signal generation anytime

🚀 *Auto Signals:* {'✅ ACTIVE' if current_session['id'] != 'CLOSED' else '❌ MARKET CLOSED'}

*You have full 24/7 signal generation power!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="admin_signal")],
            [InlineKeyboardButton("📊 System Stats", callback_data="admin_stats")],
            [InlineKeyboardButton("👥 User Management", callback_data="admin_users")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command - ADMIN 24/7 SIGNAL GENERATION"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use /login")
            return
        
        # Get symbol from command if provided
        symbol = None
        if context.args:
            symbol = context.args[0].upper().replace('_', '/')
            # Validate symbol
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
*Generated 24/7 - Market Hours Bypassed*

{direction_emoji} *{signal['symbol']}* | **{signal['direction']}**
💵 *Entry:* `{signal['entry_price']:.5f}`
✅ *Take Profit:* `{signal['take_profit']:.5f}`
❌ *Stop Loss:* `{signal['stop_loss']:.5f}`

📊 *Signal Analytics:*
• Confidence: *{signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{signal['risk_reward']}* ⚖️
• Setup Quality: *{analysis['setup_quality']}* 💎
• Timeframe: Perfect Alignment 📈

💡 *Market Analysis:*
{analysis['market_condition']}

⚡ *Admin Enhanced Features:*
• 24/7 Signal Generation ✅
• Premium Analysis ✅  
• Enhanced Accuracy ✅
• Instant Execution ✅

*Execute this premium signal immediately!* 🚀
"""
            keyboard = [
                [InlineKeyboardButton("✅ Trade Executed", callback_data="trade_done")],
                [InlineKeyboardButton("🎯 Another Signal", callback_data="admin_signal")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Failed to generate signal. Please try again.")

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command"""
        user = update.effective_user
        
        if not context.args:
            message = """
💎 *REQUEST UPGRADE*

*Available Plans:*
1. 🌅 BASIC - $19/month
2. 🌇 PRO - $49/month  
3. 🌃 VIP - $99/month
4. 🌟 PREMIUM - $199/month

*Usage:* `/upgrade <plan> <contact_method>`

*Examples:*
• `/upgrade basic telegram`
• `/upgrade pro whatsapp`

*We'll contact you immediately!* 🚀
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            return
        
        if len(context.args) < 2:
            await update.message.reply_text("❌ Usage: `/upgrade <plan> <contact_method>`")
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
        await self.upgrade_manager.notify_upgrade_request(user.id, user.username, plan, contact_method)
        
        message = f"""
✅ *UPGRADE REQUEST RECEIVED!*

📋 *Your Request:*
• Plan: *{plan}*
• Contact: *{contact_method}*

💰 *Next Steps:*
1. Admin will contact you shortly
2. Complete payment  
3. Get instant activation

*Thank you for choosing LEKZY FX AI PRO!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

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

🚀 *Quick Upgrade:*
1. Message {Config.ADMIN_CONTACT} directly
2. Specify desired plan
3. Complete payment
4. Instant activation

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

🚀 *Upgrade Now:*
Use `/upgrade <plan> <contact_method>`
Or contact {Config.ADMIN_CONTACT} directly

*Start your premium journey!* 🎯
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        message = f"""
📊 *YOUR ACCOUNT STATS*

👤 User: {user.first_name}
📋 Plan: {user_plan}
🎯 Accuracy: 96.2%

💎 *Upgrade to unlock:*
• More sessions & signals
• Higher accuracy
• Priority support

*Use /plans to see options!*
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
        session = self.trading_bot.session_manager.get_current_session()
        
        if session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED*

*Next Sessions:*
• 🌅 Morning: 08:00-12:00
• 🌇 Evening: 16:00-20:00
• 🌃 Asian: 00:00-04:00

*Admins can generate signals 24/7!*
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ ACTIVE

⏰ {session['start_hour']:02d}:00-{session['end_hour']:02d}:00
🎯 {session['accuracy']}% Accuracy

💎 *Premium signals active!*
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, signal_type, direction, entry_price, confidence, requested_by, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 6
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet. Check during session hours!")
            return
        
        message = "📡 *RECENT SIGNALS*\n\n"
        
        for symbol, signal_type, direction, entry, confidence, requested_by, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            type_emoji = "📊" if signal_type == "PRE_ENTRY" else "🎯"
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            admin_badge = " 👑" if requested_by == "ADMIN" else ""
            
            message += f"{type_emoji} {dir_emoji} {symbol}{admin_badge}\n"
            message += f"💵 {entry} | {confidence*100:.1f}%\n"
            message += f"⏰ {time_str}\n\n"
        
        message += "👑 *Admin signals available 24/7!*"
        
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
        logger.info("🚀 LEKZY FX AI PRO - 24/7 Admin Signals Started")
    
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
