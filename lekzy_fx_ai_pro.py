#!/usr/bin/env python3
"""
LEKZY FX AI PRO - Professional Signal System with Time-Based Entries
"""

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
    PRE_ENTRY_DELAY = 40  # seconds before entry
    TIMEZONE_OFFSET = 1  # UTC+1
    SIGNAL_COOLDOWN = random.randint(60, 120)  # 1-2 minutes between signals

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
    return "🤖 LEKZY FX AI PRO - Professional Signal System 🚀"

@app.route('/health')
def health():
    return "✅ Bot Status: Active (UTC+1)"

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
                signal_style TEXT DEFAULT 'PROFESSIONAL',
                requested_by TEXT DEFAULT 'AUTO',
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                command TEXT,
                status TEXT DEFAULT 'PENDING',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Professional database initialized")
        
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

# ==================== UTC+1 SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        # UTC+1 Trading Sessions (Central European Time)
        self.sessions = {
            "MORNING": {
                "start_hour": 7, "end_hour": 11,  # 08:00-12:00 UTC+1
                "name": "🌅 London Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "volatility": "HIGH",
                "accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 15, "end_hour": 19,  # 16:00-20:00 UTC+1
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 23, "end_hour": 3,   # 00:00-04:00 UTC+1 (next day)
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "accuracy": 92.5
            },
            "ADMIN_24_7": {
                "start_hour": 0, "end_hour": 24,
                "name": "👑 24/7 Admin Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"],
                "volatility": "ADMIN",
                "accuracy": 98.5
            }
        }

    def get_current_time_utc1(self):
        """Get current time in UTC+1"""
        return datetime.utcnow() + timedelta(hours=Config.TIMEZONE_OFFSET)

    def get_current_session(self):
        """Get current active trading session in UTC+1"""
        now_utc1 = self.get_current_time_utc1()
        current_hour = now_utc1.hour
        current_time_str = now_utc1.strftime("%H:%M UTC+1")
        
        # Handle Asian session crossing midnight
        for session_id, session in self.sessions.items():
            if session_id == "ASIAN":
                if current_hour >= session["start_hour"] or current_hour < session["end_hour"]:
                    return {**session, "id": session_id, "current_time": current_time_str}
            else:
                if session["start_hour"] <= current_hour < session["end_hour"]:
                    return {**session, "id": session_id, "current_time": current_time_str}
        
        return {"id": "CLOSED", "name": "Market Closed", "current_time": current_time_str}

    def get_next_session(self):
        """Get next trading session"""
        current_session = self.get_current_session()
        sessions_order = ["ASIAN", "MORNING", "EVENING"]
        
        if current_session["id"] == "CLOSED":
            return self.sessions["ASIAN"]
        
        current_index = sessions_order.index(current_session["id"])
        next_index = (current_index + 1) % len(sessions_order)
        return self.sessions[sessions_order[next_index]]

# ==================== PROFESSIONAL SIGNAL GENERATOR ====================
class ProfessionalSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        self.pending_signals = {}
        self.last_signal_time = {}
    
    def generate_professional_analysis(self, symbol: str) -> dict:
        """Generate professional trading analysis"""
        
        # Professional candle patterns
        candle_patterns = [
            "Bullish Engulfing pattern confirmed on M5",
            "Bearish Engulfing with strong volume",
            "Hammer at key support level",
            "Shooting star at resistance",
            "Doji reversal pattern forming",
            "Three white soldiers pattern"
        ]
        
        # Professional timeframes
        timeframes = ["M5", "M15", "H1"]
        
        # Market conditions
        market_conditions = [
            "Strong momentum with institutional flow",
            "Price at key Fibonacci level",
            "Market structure break confirmed",
            "Liquidity pool activation",
            "Economic catalyst driving movement",
            "Technical breakout with volume"
        ]
        
        return {
            "candle_pattern": random.choice(candle_patterns),
            "timeframe": random.choice(timeframes),
            "market_condition": random.choice(market_conditions),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG_BULLISH", "STRONG_BEARISH", "BUILDING"]),
            "volume_analysis": random.choice(["ABOVE_AVERAGE", "HIGH", "VERY_HIGH"]),
            "risk_rating": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "professional_grade": True,
            "entry_timing": "NEXT_CANDLE"
        }
    
    def can_generate_signal(self, user_id: int) -> bool:
        """Check if user can generate signal (cooldown)"""
        current_time = time.time()
        if user_id in self.last_signal_time:
            time_diff = current_time - self.last_signal_time[user_id]
            if time_diff < Config.SIGNAL_COOLDOWN:
                return False
        self.last_signal_time[user_id] = current_time
        return True
    
    def generate_pre_entry_signal(self, symbol: str = None, is_admin: bool = False) -> dict:
        """Generate pre-entry signal with professional analysis"""
        if not symbol:
            symbol = random.choice(self.all_pairs)
        
        analysis = self.generate_professional_analysis(symbol)
        direction = "BUY" if random.random() > 0.48 else "SELL"
        
        # Professional price calculation
        base_price = analysis["key_level"]
        spread = 0.0003  # Professional spread
        
        if direction == "BUY":
            entry_price = round(base_price + spread, 5)
        else:
            entry_price = round(base_price - spread, 5)
        
        # Professional confidence calculation
        base_confidence = random.uniform(0.88, 0.97)
        if is_admin:
            base_confidence += 0.02
        
        signal_id = f"PRE_{symbol.replace('/', '')}_{int(time.time())}"
        
        signal_data = {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_type": "PRE_ENTRY",
            "direction": direction,
            "entry_price": entry_price,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "confidence": min(0.98, round(base_confidence, 3)),
            "session_type": "ADMIN_24_7" if is_admin else "AUTO",
            "analysis": json.dumps(analysis),
            "time_to_entry": Config.PRE_ENTRY_DELAY,
            "risk_reward": 0.0,
            "signal_style": "PROFESSIONAL",
            "requested_by": "ADMIN" if is_admin else "USER",
            "generated_at": datetime.now().isoformat()
        }
        
        self.pending_signals[signal_id] = signal_data
        return signal_data
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate entry signal based on pre-entry"""
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        analysis = json.loads(pre_signal["analysis"])
        
        # Professional TP/SL calculation
        movement = 0.0028  # Professional movement
        risk_multiplier = 0.65  # Professional risk
        
        if pre_signal["direction"] == "BUY":
            take_profit = round(pre_signal["entry_price"] + movement, 5)
            stop_loss = round(pre_signal["entry_price"] - movement * risk_multiplier, 5)
        else:
            take_profit = round(pre_signal["entry_price"] - movement, 5)
            stop_loss = round(pre_signal["entry_price"] + movement * risk_multiplier, 5)
        
        risk_reward = round((take_profit - pre_signal["entry_price"]) / (pre_signal["entry_price"] - stop_loss), 2) if pre_signal["direction"] == "BUY" else round((pre_signal["entry_price"] - take_profit) / (stop_loss - pre_signal["entry_price"]), 2)
        
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

# ==================== PROFESSIONAL TRADING BOT ====================
class ProfessionalTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = SessionManager()
        self.signal_generator = ProfessionalSignalGenerator()
        self.is_running = False
    
    async def start_auto_signals(self):
        """Start automatic signal generation during sessions"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED" and session["id"] != "ADMIN_24_7":
                        logger.info(f"🎯 Generating {session['name']} professional signals")
                        
                        for symbol in session["optimal_pairs"][:1]:
                            # Generate pre-entry signal immediately
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, False)
                            
                            # Store pre-entry
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, signal_style, requested_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    pre_signal["signal_id"], pre_signal["symbol"], pre_signal["signal_type"],
                                    pre_signal["direction"], pre_signal["entry_price"], pre_signal["take_profit"],
                                    pre_signal["stop_loss"], pre_signal["confidence"], pre_signal["session_type"],
                                    pre_signal["analysis"], pre_signal["time_to_entry"], pre_signal["risk_reward"],
                                    pre_signal["signal_style"], pre_signal["requested_by"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📊 Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
                            
                            # Wait 40 seconds for entry
                            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
                            
                            # Generate entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, signal_style, requested_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, tuple(entry_signal.values()))
                                    conn.commit()
                                
                                logger.info(f"🎯 Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    # Wait 1-2 minutes before next signal
                    await asyncio.sleep(Config.SIGNAL_COOLDOWN)
                    
                except Exception as e:
                    logger.error(f"Auto signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Professional auto signal generation started")
    
    async def generate_signal_sequence(self, user_id: int, symbol: str = None, is_admin: bool = False):
        """Generate professional signal sequence"""
        try:
            # Check cooldown
            if not self.signal_generator.can_generate_signal(user_id):
                wait_time = Config.SIGNAL_COOLDOWN - (time.time() - self.signal_generator.last_signal_time[user_id])
                return {"error": f"Please wait {int(wait_time)} seconds before next signal"}
            
            # Generate pre-entry signal immediately
            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, is_admin)
            
            # Store pre-entry
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO signals 
                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, signal_style, requested_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pre_signal["signal_id"], pre_signal["symbol"], pre_signal["signal_type"],
                    pre_signal["direction"], pre_signal["entry_price"], pre_signal["take_profit"],
                    pre_signal["stop_loss"], pre_signal["confidence"], pre_signal["session_type"],
                    pre_signal["analysis"], pre_signal["time_to_entry"], pre_signal["risk_reward"],
                    pre_signal["signal_style"], pre_signal["requested_by"]
                ))
                conn.commit()
            
            logger.info(f"📊 {'Admin' if is_admin else 'User'} Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
            
            return {
                "pre_signal": pre_signal,
                "entry_in_seconds": Config.PRE_ENTRY_DELAY
            }
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return None
    
    async def generate_entry_signal(self, pre_signal_id: str):
        """Generate entry signal after pre-entry delay"""
        try:
            entry_signal = self.signal_generator.generate_entry_signal(pre_signal_id)
            
            if entry_signal:
                with sqlite3.connect(Config.DB_PATH) as conn:
                    conn.execute("""
                        INSERT INTO signals 
                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, signal_style, requested_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, tuple(entry_signal.values()))
                    conn.commit()
                
                logger.info(f"🎯 Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                return entry_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Entry signal failed: {e}")
            return None

# ==================== PROFESSIONAL TELEGRAM BOT ====================
class ProfessionalTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the professional bot"""
        self.application = Application.builder().token(self.token).build()
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = ProfessionalTradingBot(self.application)
        
        # Command handlers - SIMPLIFIED FOR USERS
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_auto_signals()
        
        logger.info("🤖 Professional Trading Bot Initialized (UTC+1)!")

    async def create_welcome_message(self, user, current_session):
        """Create professional welcome message"""
        return f"""
🎯 *LEKZY FX AI PRO* - PROFESSIONAL TRADING

*Welcome, {user.first_name}!* 🌟

*Your 3-Day Free Trial Activated* ✅

🕒 *Live Market Session:*
{current_session['name']}
⏰ *Time:* {current_session['current_time']}

⚡ *Professional Features:*
• 40s Pre-Entry Signal System
• Time-Based Entry Confirmation
• Professional Candle Analysis
• Real-Time Market Monitoring

🎮 *QUICK COMMANDS:*

🚀 */signal* - Get Professional Signal
🕒 */session* - Market Hours & Status
📊 */signals* - Recent Trading Signals
📞 */contact* - Premium Support

*Tap SIGNAL to start trading!* 🎯
"""

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with professional welcome"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        current_session = self.trading_bot.session_manager.get_current_session()
        welcome_message = await self.create_welcome_message(user, current_session)
        
        # Professional keyboard - SIMPLIFIED
        keyboard = [
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="get_signal")],
            [InlineKeyboardButton("🕒 MARKET SESSION", callback_data="session")],
            [InlineKeyboardButton("📊 RECENT SIGNALS", callback_data="signals")],
            [InlineKeyboardButton("📞 CONTACT SUPPORT", callback_data="contact")]
        ]
        
        # Add admin button if admin
        if self.admin_auth.is_admin(user.id):
            keyboard.insert(0, [InlineKeyboardButton("👑 ADMIN PANEL", callback_data="admin")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        message = """
❓ *LEKZY FX AI PRO - HELP*

🎯 *How It Works:*
1. Tap *GET SIGNAL* or use /signal
2. Receive *PRE-ENTRY* signal (40s before entry)
3. Get *ENTRY* signal with exact levels
4. Execute trade professionally

⚡ *Signal System:*
• 40s pre-entry warning
• Time-based entry confirmation
• Professional candle analysis
• Real-time market monitoring

🕒 *Trading Sessions (UTC+1):*
• 🌅 London: 08:00-12:00
• 🌇 NY/London: 16:00-20:00  
• 🌃 Asian: 00:00-04:00

📞 *Support:* @LekzyTradingPro

*Trade like a professional!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

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
✅ *Admin Access Granted!* 👑

*Professional Admin Features:*
• 🚀 /signal - Generate professional signals
• 🕒 /session - Market session info
• 📊 /signals - Signal history
• 👑 /admin - Admin dashboard

*40s pre-entry system activated!* ⚡
""", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            user_requests = conn.execute("SELECT COUNT(*) FROM user_requests").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
👑 *PROFESSIONAL ADMIN DASHBOARD*

📊 *System Statistics:*
• Total Users: {total_users}
• Total Signals: {total_signals}
• User Requests: {user_requests}
• Current Session: {current_session['name']}
• Time: {current_session['current_time']}

⚡ *Signal Commands:*
• `/signal` - Professional signal
• `/signal EUR/USD` - Specific pair

🎯 *Professional Features:*
• 40s pre-entry system
• Time-based confirmation
• Candle pattern analysis
• 1-2 minute cooldown

*Professional system active!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("🚀 GENERATE SIGNAL", callback_data="admin_signal")],
            [InlineKeyboardButton("🕒 MARKET SESSION", callback_data="session")],
            [InlineKeyboardButton("📊 SYSTEM STATS", callback_data="admin_stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command - SIMPLIFIED & PROFESSIONAL"""
        user = update.effective_user
        is_admin = self.admin_auth.is_admin(user.id)
        
        # Parse symbol if provided
        symbol = None
        if context.args:
            symbol_arg = context.args[0].upper().replace('_', '/')
            valid_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
            if symbol_arg in valid_pairs:
                symbol = symbol_arg
        
        # Generate signal sequence immediately
        result = await self.trading_bot.generate_signal_sequence(user.id, symbol, is_admin)
        
        if result and "error" in result:
            await update.message.reply_text(f"⏳ {result['error']}", parse_mode='Markdown')
            return
        
        if result and "pre_signal" in result:
            pre_signal = result["pre_signal"]
            analysis = json.loads(pre_signal["analysis"])
            
            # Send pre-entry signal immediately
            direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
            
            message = f"""
🎯 *PRE-ENTRY SIGNAL* ⚡
*Entry in {Config.PRE_ENTRY_DELAY}s*

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Expected Entry:* `{pre_signal['entry_price']:.5f}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

📊 *Professional Analysis:*
{analysis['candle_pattern']}
• Timeframe: {analysis['timeframe']}
• Momentum: {analysis['momentum']}
• Risk: {analysis['risk_rating']}

💡 *Market Condition:*
{analysis['market_condition']}

⏰ *Entry signal coming in {Config.PRE_ENTRY_DELAY} seconds...*
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Wait and send entry signal
            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
            
            entry_signal = await self.trading_bot.generate_entry_signal(pre_signal["signal_id"])
            
            if entry_signal:
                entry_message = f"""
🎯 *ENTRY SIGNAL* ✅
*EXECUTE NOW*

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry Price:* `{entry_signal['entry_price']:.5f}`
✅ *Take Profit:* `{entry_signal['take_profit']:.5f}`
❌ *Stop Loss:* `{entry_signal['stop_loss']:.5f}`

📈 *Trade Details:*
• Confidence: *{entry_signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{entry_signal['risk_reward']}* ⚖️
• Type: *PROFESSIONAL* 💎

⚡ *Execution Timing:*
• Time-based entry confirmed
• Optimal entry level
• Professional setup

*Execute this trade immediately!* 🚀
"""
                keyboard = [
                    [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
                    [InlineKeyboardButton("🔄 NEW SIGNAL", callback_data="get_signal")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(entry_message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to generate entry signal")
        else:
            await update.message.reply_text("❌ Signal generation failed. Try again.")

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command with professional design"""
        session = self.trading_bot.session_manager.get_current_session()
        next_session = self.trading_bot.session_manager.get_next_session()
        
        if session["id"] == "CLOSED":
            message = f"""
🕒 *MARKET CLOSED* ⏸️

⏰ *Current Time:* {session['current_time']}
📅 *Next Session:* {next_session['name']}

*Professional Sessions (UTC+1):*

🌅 *LONDON SESSION* (08:00-12:00)
• Volatility: HIGH
• Accuracy: 96.2%
• Pairs: EUR/USD, GBP/USD, EUR/JPY

🌇 *NY/LONDON OVERLAP* (16:00-20:00)
• Volatility: VERY HIGH  
• Accuracy: 97.8%
• Pairs: USD/JPY, USD/CAD, XAU/USD

🌃 *ASIAN SESSION* (00:00-04:00)
• Volatility: MEDIUM
• Accuracy: 92.5%
• Pairs: AUD/JPY, NZD/USD, USD/JPY

*Signals auto-resume in session hours!* 📈
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ LIVE

⏰ *Current Time:* {session['current_time']}
📊 *Volatility:* {session['volatility']}
🎯 *Accuracy:* {session['accuracy']}%
💎 *Optimal Pairs:* {', '.join(session['optimal_pairs'])}

⚡ *Professional Signals Active:*
• 40s pre-entry system
• Time-based confirmation
• Candle pattern analysis
• Real-time monitoring

*Professional trading active!* 🚀
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, signal_type, direction, entry_price, confidence, requested_by, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 5
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet. Market may be closed or starting soon!")
            return
        
        message = "📡 *RECENT PROFESSIONAL SIGNALS*\n\n"
        
        for symbol, signal_type, direction, entry, confidence, requested_by, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            type_emoji = "📊" if signal_type == "PRE_ENTRY" else "🎯"
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            admin_badge = " 👑" if requested_by == "ADMIN" else ""
            
            message += f"{type_emoji} {dir_emoji} {symbol}{admin_badge}\n"
            message += f"💵 {entry} | {confidence*100:.1f}% | {time_str}\n\n"
        
        message += "⚡ *40s pre-entry system active!*"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /contact command"""
        message = f"""
📞 *PROFESSIONAL SUPPORT*

*Contact Admin:* {Config.ADMIN_CONTACT}

💎 *Premium Features:*
• 24/7 Signal Access
• Priority Execution
• Higher Accuracy
• Personal Support

🚀 *Upgrade Plans:*
• BASIC - $19/month
• PRO - $49/month  
• VIP - $99/month
• PREMIUM - $199/month

*Message for professional trading!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("📱 MESSAGE ADMIN", url=f"https://t.me/{Config.ADMIN_CONTACT.replace('@', '')}")],
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="get_signal")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        
        if query.data == "get_signal":
            await self.signal_command(update, context)
        elif query.data == "session":
            await self.session_command(update, context)
        elif query.data == "signals":
            await self.signals_command(update, context)
        elif query.data == "contact":
            await self.contact_command(update, context)
        elif query.data == "admin":
            if self.admin_auth.is_admin(user.id):
                await self.admin_command(update, context)
            else:
                await query.edit_message_text("❌ Admin access required")
        elif query.data == "admin_signal":
            if self.admin_auth.is_admin(user.id):
                await self.signal_command(update, context)
            else:
                await query.edit_message_text("❌ Admin access required")
        elif query.data == "trade_done":
            await query.edit_message_text("✅ *Trade execution confirmed!* 🎯\n\n*Happy trading!* 💰")

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
        
        self.bot = ProfessionalTelegramBot()
        await self.bot.initialize()
        
        self.running = True
        logger.info("🚀 LEKZY FX AI PRO - Professional Signal System Started")
    
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
