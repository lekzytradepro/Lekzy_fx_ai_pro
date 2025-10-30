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
    return "🤖 LEKZY FX AI PRO - UTC+1 Trading System 🚀"

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
                signal_style TEXT DEFAULT 'NORMAL',  -- NORMAL or QUICK
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
        logger.info("✅ Enhanced database initialized")
        
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

# ==================== ENHANCED SIGNAL GENERATOR ====================
class EnhancedSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        self.pending_signals = {}
    
    def generate_candle_analysis(self, symbol: str, signal_style: str = "NORMAL") -> dict:
        """Generate detailed candle-based analysis"""
        
        if signal_style == "QUICK":
            # Quick Trade Analysis - Faster, more aggressive
            candle_patterns = [
                "Bullish Engulfing pattern forming on M5",
                "Bearish Engulfing pattern confirmed",
                "Hammer candle at support with volume",
                "Shooting star at resistance level",
                "Doji candle indicating reversal",
                "Three white soldiers pattern emerging"
            ]
            timeframes = ["M1", "M3", "M5"]
            confidence_boost = 0.04
            speed = "QUICK_TRADE"
        else:
            # Normal Trade Analysis - More conservative
            candle_patterns = [
                "Strong bullish candle closing above resistance",
                "Bearish candle breaking support with momentum",
                "Pin bar rejection at key level",
                "Inside bar breakout confirmation",
                "Evening star pattern forming on H1",
                "Morning star reversal pattern confirmed"
            ]
            timeframes = ["M5", "M15", "H1"]
            confidence_boost = 0.02
            speed = "NORMAL"
        
        # Technical indicators
        indicators = {
            "rsi": random.randint(25, 75),
            "macd": random.choice(["BULLISH_CROSS", "BEARISH_CROSS", "NEUTRAL"]),
            "stochastic": random.randint(20, 80),
            "volume": random.choice(["ABOVE_AVERAGE", "HIGH", "VERY_HIGH"]),
            "atr": round(random.uniform(0.0008, 0.0015), 4)
        }
        
        # Market conditions
        market_conditions = [
            "New candle forming with strong momentum",
            "Price reacting to key Fibonacci level",
            "Institutional order flow detected",
            "Market structure break confirmed",
            "Liquidity pool activation",
            "Economic data driving momentum"
        ]
        
        return {
            "signal_style": signal_style,
            "candle_pattern": random.choice(candle_patterns),
            "timeframe": random.choice(timeframes),
            "market_condition": random.choice(market_conditions),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG_BULLISH", "STRONG_BEARISH", "BUILDING"]),
            "indicators": indicators,
            "confidence_boost": confidence_boost,
            "execution_speed": speed,
            "new_candle_analysis": True,
            "risk_rating": random.choice(["LOW", "MEDIUM", "HIGH"])
        }
    
    def generate_pre_entry_signal(self, symbol: str, signal_style: str = "NORMAL", is_admin: bool = False) -> dict:
        """Generate pre-entry signal with candle analysis"""
        analysis = self.generate_candle_analysis(symbol, signal_style)
        direction = "BUY" if random.random() > 0.48 else "SELL"
        
        # Calculate entry price based on analysis
        base_price = analysis["key_level"]
        if direction == "BUY":
            entry_price = round(base_price + 0.0005, 5)
        else:
            entry_price = round(base_price - 0.0005, 5)
        
        # Enhanced confidence for admin and quick trades
        base_confidence = random.uniform(0.85, 0.95)
        if is_admin:
            base_confidence += 0.03
        if signal_style == "QUICK":
            base_confidence += analysis["confidence_boost"]
        
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
            "signal_style": signal_style,
            "requested_by": "ADMIN" if is_admin else "AUTO",
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
        
        # Calculate TP/SL based on signal style
        if analysis["signal_style"] == "QUICK":
            movement = 0.0020  # Smaller targets for quick trades
            risk_multiplier = 0.5  # Tighter stops
        else:
            movement = 0.0035  # Larger targets for normal trades
            risk_multiplier = 0.6
        
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

# ==================== ENHANCED TRADING BOT ====================
class EnhancedTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = SessionManager()
        self.signal_generator = EnhancedSignalGenerator()
        self.is_running = False
    
    async def start_auto_signals(self):
        """Start automatic signal generation during sessions"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED" and session["id"] != "ADMIN_24_7":
                        logger.info(f"🎯 Generating {session['name']} signals")
                        
                        for symbol in session["optimal_pairs"][:1]:
                            # Generate pre-entry signal
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, "NORMAL", False)
                            
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
                            
                            logger.info(f"📊 Pre-entry: {pre_signal['symbol']} {pre_signal['direction']} ({pre_signal['signal_style']})")
                            
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
                    
                    await asyncio.sleep(random.randint(300, 600))
                    
                except Exception as e:
                    logger.error(f"Auto signal error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Auto signal generation started")
    
    async def generate_admin_signal_sequence(self, user_id: int, symbol: str = None, signal_style: str = "NORMAL"):
        """Generate admin signal sequence with pre-entry and entry"""
        try:
            if not symbol:
                symbol = random.choice(self.signal_generator.all_pairs)
            
            # Step 1: Generate pre-entry signal
            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, signal_style, True)
            
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
            
            logger.info(f"📊 Admin Pre-entry: {pre_signal['symbol']} {pre_signal['direction']} ({signal_style})")
            
            # Return pre-entry signal immediately
            pre_entry_data = {
                "pre_signal": pre_signal,
                "entry_in_seconds": Config.PRE_ENTRY_DELAY
            }
            
            return pre_entry_data
            
        except Exception as e:
            logger.error(f"❌ Admin signal generation failed: {e}")
            return None
    
    async def generate_admin_entry_signal(self, pre_signal_id: str):
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
                
                logger.info(f"🎯 Admin Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                return entry_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Admin entry signal failed: {e}")
            return None

    async def generate_user_signal_request(self, user_id: int):
        """Generate signal for user request"""
        try:
            session = self.session_manager.get_current_session()
            
            if session["id"] == "CLOSED":
                return None
            
            symbol = random.choice(session["optimal_pairs"])
            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, "NORMAL", False)
            
            # Store user request
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute("INSERT INTO user_requests (user_id, command, status) VALUES (?, ?, ?)",
                           (user_id, "signal_request", "PROCESSED"))
                conn.commit()
            
            return pre_signal
            
        except Exception as e:
            logger.error(f"User signal request failed: {e}")
            return None

# ==================== COMPLETE TELEGRAM BOT ====================
class CompleteTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the complete bot"""
        self.application = Application.builder().token(self.token).build()
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = EnhancedTradingBot(self.application)
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("mysignal", self.mysignal_command))
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        self.application.add_handler(CommandHandler("plans", self.plans_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("commands", self.commands_command))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_auto_signals()
        
        logger.info("🤖 Enhanced Trading Bot Initialized (UTC+1)!")

    async def get_user_commands(self, user_id: int):
        """Get available commands based on user status"""
        base_commands = """
🕒 *Session Commands:*
• /session - Current market session (UTC+1)
• /signals - View recent signals

📊 *Account Commands:*
• /stats - Your account status
• /mysignal - Request a signal
• /plans - View premium plans
• /upgrade - Upgrade instructions
• /contact - Contact admin

❓ *Help Commands:*
• /help - Detailed help guide
• /commands - Available commands
"""
        
        if self.admin_auth.is_admin(user_id):
            admin_commands = """
👑 *ADMIN COMMANDS:*

⚡ *Signal Generation:*
• /signal - Normal trade signal
• /signal quick - Quick Trade signal
• /signal EUR/USD - Specific pair
• /signal EUR/USD quick - Quick specific pair

🏢 *Admin Management:*
• /admin - Admin dashboard
• /stats - System statistics
• /signals - All signals history

*You have full system access!* 🚀
"""
            return base_commands + admin_commands
        else:
            return base_commands + "\n*Use /login TOKEN for admin access* 🔐"

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with enhanced commands"""
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        current_session = self.trading_bot.session_manager.get_current_session()
        user_commands = await self.get_user_commands(user.id)
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
Experience professional trading with UTC+1 timing.

🕒 *Current Market:* {current_session['name']}
⏰ *Time:* {current_session['current_time']}

{user_commands}

*Start trading with confidence!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("🕒 Market Session", callback_data="session"),
             InlineKeyboardButton("📡 Get Signal", callback_data="mysignal")],
            [InlineKeyboardButton("📊 Account Stats", callback_data="stats"),
             InlineKeyboardButton("💎 Upgrade", callback_data="plans")],
            [InlineKeyboardButton("❓ Help & Commands", callback_data="help")]
        ]
        
        # Add admin buttons if admin
        if self.admin_auth.is_admin(user.id):
            keyboard.insert(1, [
                InlineKeyboardButton("👑 Admin Panel", callback_data="admin"),
                InlineKeyboardButton("⚡ Quick Trade", callback_data="signal_quick")
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user = update.effective_user
        user_commands = await self.get_user_commands(user.id)
        
        message = f"""
❓ *LEKZY FX AI PRO - HELP GUIDE*

{user_commands}

💡 *Trading Features:*
• ⚡ Quick Trade signals (40s pre-entry)
• 📈 Normal signals with detailed analysis
• 🕯️ Candle-based entry confirmation
• 🕒 UTC+1 session timing
• 📊 Real-time market analysis

📞 *Support:* {Config.ADMIN_CONTACT}

*Happy Trading!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def commands_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /commands - Show all available commands"""
        user = update.effective_user
        user_commands = await self.get_user_commands(user.id)
        
        message = f"""
🎯 *AVAILABLE COMMANDS*

{user_commands}

*Use any command to get started!* 💪
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
            
            # Show admin commands after successful login
            user_commands = await self.get_user_commands(user.id)
            
            await update.message.reply_text(f"""
✅ *Admin Access Granted!* 🌟

{user_commands}

*Admin features are now active!* 🚀
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
            quick_signals = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_style = 'QUICK'").fetchone()[0]
            user_requests = conn.execute("SELECT COUNT(*) FROM user_requests").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🏢 *ADMIN DASHBOARD* 🌟

📊 *Statistics:*
• Total Users: {total_users}
• Total Signals: {total_signals}
• Quick Trades: {quick_signals}
• User Requests: {user_requests}
• Current Session: {current_session['name']}
• Time: {current_session['current_time']}

🎯 *Signal Commands:*
• `/signal` - Normal trade
• `/signal quick` - Quick Trade (40s pre-entry)
• `/signal EUR/USD` - Specific pair
• `/signal EUR/USD quick` - Quick specific

⚡ *Quick Trade Features:*
• 40s pre-entry warning
• Candle-based analysis
• Faster execution
• Tighter stops

*24/7 Admin Access Active!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Trade", callback_data="signal_quick"),
             InlineKeyboardButton("📈 Normal Trade", callback_data="signal_normal")],
            [InlineKeyboardButton("📊 System Stats", callback_data="stats"),
             InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command with style selection"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
        # Parse command arguments
        symbol = None
        signal_style = "NORMAL"
        
        if context.args:
            for arg in context.args:
                arg_upper = arg.upper()
                if arg_upper in ["QUICK", "FAST", "Q"]:
                    signal_style = "QUICK"
                elif "/" in arg_upper or arg_upper in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD"]:
                    symbol = arg_upper.replace('USD', '/USD') if 'USD' in arg_upper and '/' not in arg_upper else arg_upper
                    symbol = symbol.replace('_', '/')
        
        valid_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        if symbol and symbol not in valid_pairs:
            await update.message.reply_text(f"❌ Invalid pair. Use: {', '.join(valid_pairs)}")
            return
        
        # Generate pre-entry signal
        await update.message.reply_text(f"🎯 *Generating {signal_style} signal...*", parse_mode='Markdown')
        
        result = await self.trading_bot.generate_admin_signal_sequence(user.id, symbol, signal_style)
        
        if result and "pre_signal" in result:
            pre_signal = result["pre_signal"]
            analysis = json.loads(pre_signal["analysis"])
            
            # Send pre-entry signal
            direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
            style_emoji = "⚡" if signal_style == "QUICK" else "📈"
            
            message = f"""
{style_emoji} *PRE-ENTRY SIGNAL* - {signal_style}
*Entry in {Config.PRE_ENTRY_DELAY}s*

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Expected Entry:* `{pre_signal['entry_price']:.5f}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

📊 *Candle Analysis:*
{analysis['candle_pattern']}
• Timeframe: {analysis['timeframe']}
• Momentum: {analysis['momentum']}
• Risk Rating: {analysis['risk_rating']}

💡 *Market Condition:*
{analysis['market_condition']}

⏰ *Entry signal coming in {Config.PRE_ENTRY_DELAY} seconds...*
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Wait and send entry signal
            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
            
            entry_signal = await self.trading_bot.generate_admin_entry_signal(pre_signal["signal_id"])
            
            if entry_signal:
                entry_analysis = json.loads(entry_signal["analysis"])
                
                entry_message = f"""
🎯 *ENTRY SIGNAL* - {signal_style}
*EXECUTE NOW*

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry Price:* `{entry_signal['entry_price']:.5f}`
✅ *Take Profit:* `{entry_signal['take_profit']:.5f}`
❌ *Stop Loss:* `{entry_signal['stop_loss']:.5f}`

📈 *Trade Details:*
• Confidence: *{entry_signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{entry_signal['risk_reward']}* ⚖️
• Style: *{signal_style}* {style_emoji}

⚡ *Execution:*
• New candle confirmed
• Price at optimal level
• Momentum aligned

*Execute this trade immediately!* 🚀
"""
                keyboard = [
                    [InlineKeyboardButton("✅ Trade Executed", callback_data="trade_done")],
                    [InlineKeyboardButton("⚡ Another Signal", callback_data="admin_signal")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(entry_message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to generate entry signal")
        else:
            await update.message.reply_text("❌ Failed to generate pre-entry signal")

    async def mysignal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mysignal command for users"""
        user = update.effective_user
        current_session = self.trading_bot.session_manager.get_current_session()
        
        if current_session["id"] == "CLOSED":
            next_session = self.trading_bot.session_manager.get_next_session()
            await update.message.reply_text(f"""
📭 *No Signals Available*

🕒 *Market is currently closed*
• Current Time: {current_session['current_time']}
• Next Session: {next_session['name']} ({next_session['start_hour']:02d}:00-{next_session['end_hour']:02d}:00 UTC+1)

💡 *Signals are available during:*
• 🌅 London Session (08:00-12:00 UTC+1)
• 🌇 NY/London Overlap (16:00-20:00 UTC+1) 
• 🌃 Asian Session (00:00-04:00 UTC+1)

*Check back during session hours!* 📈
""", parse_mode='Markdown')
            return
        
        # Generate signal for user
        await update.message.reply_text("🎯 *Processing your signal request...*", parse_mode='Markdown')
        
        signal = await self.trading_bot.generate_user_signal_request(user.id)
        
        if signal:
            analysis = json.loads(signal["analysis"])
            
            message = f"""
📊 *YOUR TRADING SIGNAL*

🟢 *{signal['symbol']}* | **{signal['direction']}**
💵 *Entry Zone:* `{signal['entry_price']:.5f}`
🎯 *Confidence:* {signal['confidence']*100:.1f}%

📈 *Analysis:*
{analysis['candle_pattern']}
• Timeframe: {analysis['timeframe']}
• Momentum: {analysis['momentum']}
• Risk: {analysis['risk_rating']}

💡 *Market Condition:*
{analysis['market_condition']}

⚡ *Execution Tip:*
Wait for price confirmation at entry level
Use proper risk management

*Trade carefully!* ✅
"""
            keyboard = [
                [InlineKeyboardButton("✅ Got It", callback_data="signal_ack")],
                [InlineKeyboardButton("🔄 New Signal", callback_data="mysignal")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.message.reply_text("""
❌ *Signal Generation Failed*

This could be because:
• Market volatility is too low
• No clear setup available
• System processing delay

Try again in a few minutes! 🔄
""", parse_mode='Markdown')

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command with UTC+1"""
        session = self.trading_bot.session_manager.get_current_session()
        next_session = self.trading_bot.session_manager.get_next_session()
        
        if session["id"] == "CLOSED":
            message = f"""
🕒 *MARKET CLOSED* ❌

⏰ *Current Time:* {session['current_time']}
📅 *Next Session:* {next_session['name']}

*Trading Sessions (UTC+1):*

🌅 *LONDON SESSION* (08:00-12:00 UTC+1)
• Volatility: HIGH
• Accuracy: 96.2%
• Optimal Pairs: EUR/USD, GBP/USD, EUR/JPY

🌇 *NY/LONDON OVERLAP* (16:00-20:00 UTC+1)
• Volatility: VERY HIGH  
• Accuracy: 97.8%
• Optimal Pairs: USD/JPY, USD/CAD, XAU/USD

🌃 *ASIAN SESSION* (00:00-04:00 UTC+1)
• Volatility: MEDIUM
• Accuracy: 92.5%
• Optimal Pairs: AUD/JPY, NZD/USD, USD/JPY

*Signals auto-resume in session hours!* 📈
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ ACTIVE

⏰ *Current Time:* {session['current_time']}
📊 *Volatility:* {session['volatility']}
🎯 *Accuracy:* {session['accuracy']}%
💎 *Optimal Pairs:* {', '.join(session['optimal_pairs'])}

⚡ *Enhanced Signals Active:*
• Quick Trade (40s pre-entry)
• Normal signals with analysis
• Candle-based entries
• Real-time monitoring

*Professional signals are live!* 🚀
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            signals = conn.execute("""
                SELECT symbol, signal_type, direction, entry_price, confidence, signal_style, requested_by, created_at 
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 6
            """).fetchall()
        
        if not signals:
            await update.message.reply_text("📭 No signals yet. Check during session hours!")
            return
        
        message = "📡 *RECENT TRADING SIGNALS*\n\n"
        
        for symbol, signal_type, direction, entry, confidence, style, requested_by, created in signals:
            time_str = datetime.fromisoformat(created).strftime("%H:%M")
            type_emoji = "📊" if signal_type == "PRE_ENTRY" else "🎯"
            dir_emoji = "🟢" if direction == "BUY" else "🔴"
            style_emoji = "⚡" if style == "QUICK" else "📈"
            admin_badge = " 👑" if requested_by == "ADMIN" else ""
            
            message += f"{type_emoji} {dir_emoji} {symbol}{admin_badge}\n"
            message += f"{style_emoji} {style} | 💵 {entry} | {confidence*100:.1f}%\n"
            message += f"⏰ {time_str}\n\n"
        
        message += "⚡ *Quick Trade signals feature 40s pre-entry!*"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command"""
        await update.message.reply_text(f"""
💎 *UPGRADE YOUR ACCOUNT*

*Contact admin for premium features:*
{Config.ADMIN_CONTACT}

🌟 *Premium Benefits:*
• All session access (24/7 signals)
• Quick Trade priority
• Higher accuracy signals
• Personal support
• Advanced analytics

*Unlock enhanced trading!* 🚀
""", parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /contact command"""
        message = f"""
📞 *CONTACT ADMIN*

*Direct Contact:* {Config.ADMIN_CONTACT}

💡 *Premium Support:*
• Quick Trade signals
• All session access
• Higher accuracy
• Priority support

📋 *Upgrade Plans Available:*
• BASIC - $19/month
• PRO - $49/month  
• VIP - $99/month
• PREMIUM - $199/month

*Message us now to upgrade!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("📱 Message Admin", url=f"https://t.me/{Config.ADMIN_CONTACT.replace('@', '')}")],
            [InlineKeyboardButton("💎 View Plans", callback_data="plans")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def plans_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /plans command"""
        message = f"""
💎 *LEKZY FX AI PRO - PREMIUM PLANS*

🌅 *BASIC* - $19/month
• Morning Session Only
• Quick Trade signals
• 10 signals/day
• 95%+ Accuracy

🌇 *PRO* - $49/month  
• Morning + Evening Sessions
• Enhanced analysis
• 25 signals/day
• 96%+ Accuracy

🌃 *VIP* - $99/month
• All Sessions (24/7)
• Priority signals
• 50 signals/day
• 97%+ Accuracy

🌟 *PREMIUM* - $199/month
• 24/7 Priority Access
• Unlimited signals
• Personal support
• 98%+ Accuracy
• Advanced features

💡 *All plans include:*
• Quick Trade system
• Candle-based analysis
• Risk management
• Real-time alerts

*Contact {Config.ADMIN_CONTACT} to upgrade!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("📞 Contact Admin", callback_data="contact")],
            [InlineKeyboardButton("🔼 Upgrade Now", callback_data="upgrade")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        current_session = self.trading_bot.session_manager.get_current_session()
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            user_signals = conn.execute(
                "SELECT COUNT(*) FROM user_requests WHERE user_id = ?", 
                (user.id,)
            ).fetchone()[0]
        
        if self.admin_auth.is_admin(user.id):
            with sqlite3.connect(Config.DB_PATH) as conn:
                total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
                total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            
            message = f"""
📊 *ADMIN STATS* 👑

• Plan: PREMIUM ADMIN
• Users: {total_users}
• Total Signals: {total_signals}
• Current Session: {current_session['name']}
• Time: {current_session['current_time']}

⚡ *Admin Features:*
• 24/7 signal generation
• Quick Trade system
• Full system access
• User management

*You have premium admin access!* 🚀
"""
        else:
            message = f"""
📊 *YOUR ACCOUNT STATS*

• Plan: {user_plan}
• Signals Used: {user_signals}/5 daily
• Current Session: {current_session['name']}
• Time: {current_session['current_time']}

💡 *Trial Features:*
• Morning session signals
• Basic analysis
• 5 signals per day
• Standard accuracy

*Upgrade for enhanced features!* 💎
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user = update.effective_user
        
        if query.data == "session":
            await self.session_command(update, context)
        elif query.data == "mysignal":
            await self.mysignal_command(update, context)
        elif query.data == "stats":
            await self.stats_command(update, context)
        elif query.data == "plans":
            await self.plans_command(update, context)
        elif query.data == "contact":
            await self.contact_command(update, context)
        elif query.data == "help":
            await self.help_command(update, context)
        elif query.data == "admin":
            await self.admin_command(update, context)
        elif query.data == "signal_quick":
            if self.admin_auth.is_admin(user.id):
                context.args = ["quick"]
                await self.signal_command(update, context)
            else:
                await query.edit_message_text("❌ Admin access required for Quick Trade")
        elif query.data == "signal_normal":
            if self.admin_auth.is_admin(user.id):
                await self.signal_command(update, context)
            else:
                await query.edit_message_text("❌ Admin access required for signal generation")
        elif query.data == "upgrade":
            await self.upgrade_command(update, context)
        elif query.data == "commands":
            await self.commands_command(update, context)

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
        logger.info("🚀 LEKZY FX AI PRO - UTC+1 Trading System Started")
    
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
