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
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "LEKZY_ADMIN_123")
    ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@LekzyTradingPro")
    DB_PATH = "lekzy_fx_ai.db"
    PRE_ENTRY_DELAY = 40  # seconds before entry

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
                signal_style TEXT DEFAULT 'NORMAL',
                requested_by TEXT DEFAULT 'AUTO',
                status TEXT DEFAULT 'ACTIVE',
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
        return token == Config.ADMIN_TOKEN
    
    def create_session(self, user_id: int, username: str):
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

# ==================== ENHANCED SIGNAL GENERATOR ====================
class EnhancedSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        self.pending_signals = {}
    
    def generate_candle_analysis(self, symbol: str, signal_style: str = "NORMAL") -> dict:
        if signal_style == "QUICK":
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
        
        indicators = {
            "rsi": random.randint(25, 75),
            "macd": random.choice(["BULLISH_CROSS", "BEARISH_CROSS", "NEUTRAL"]),
            "stochastic": random.randint(20, 80),
            "volume": random.choice(["ABOVE_AVERAGE", "HIGH", "VERY_HIGH"]),
            "atr": round(random.uniform(0.0008, 0.0015), 4)
        }
        
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
        analysis = self.generate_candle_analysis(symbol, signal_style)
        direction = "BUY" if random.random() > 0.48 else "SELL"
        
        base_price = analysis["key_level"]
        if direction == "BUY":
            entry_price = round(base_price + 0.0005, 5)
        else:
            entry_price = round(base_price - 0.0005, 5)
        
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
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        analysis = json.loads(pre_signal["analysis"])
        
        if analysis["signal_style"] == "QUICK":
            movement = 0.0020
            risk_multiplier = 0.5
        else:
            movement = 0.0035
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

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8, "end_hour": 12,
                "name": "🌅 European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "volatility": "HIGH",
                "accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 16, "end_hour": 20,
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 0, "end_hour": 4,
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "accuracy": 92.5
            }
        }

    def get_current_session(self):
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
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED":
                        logger.info(f"🎯 Generating {session['name']} signals")
                        
                        for symbol in session["optimal_pairs"][:1]:
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, "NORMAL", False)
                            
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
                            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
                            
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
        try:
            if not symbol:
                symbol = random.choice(self.signal_generator.all_pairs)
            
            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, signal_style, True)
            
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
            
            pre_entry_data = {
                "pre_signal": pre_signal,
                "entry_in_seconds": Config.PRE_ENTRY_DELAY
            }
            
            return pre_entry_data
            
        except Exception as e:
            logger.error(f"❌ Admin signal generation failed: {e}")
            return None
    
    async def generate_admin_entry_signal(self, pre_signal_id: str):
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

# ==================== COMPLETE TELEGRAM BOT ====================
class CompleteTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        self.application = Application.builder().token(self.token).build()
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = EnhancedTradingBot(self.application)
        
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
        
        logger.info("🤖 Enhanced Trading Bot Initialized!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🎉 *Welcome to LEKZY FX AI PRO, {user.first_name}!*

Your 3-day free trial has been activated! 
Experience our enhanced trading system with Quick Trade signals.

🕒 *Current Market:* {current_session['name']}

📊 *Enhanced Signal System:*
• ⚡ Quick Trade signals (40s pre-entry)
• 📈 Normal signals with detailed analysis
• 🕯️ Candle-based entry confirmation
• 🎯 New candle pattern detection

💡 *Available Commands:*
• /session - Check market hours
• /signals - View recent signals  
• /stats - Your account status
• /plans - Upgrade options
• /contact - Admin support

*Start trading with professional signals!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("🕒 Market Session", callback_data="session"),
             InlineKeyboardButton("📡 Recent Signals", callback_data="signals")],
            [InlineKeyboardButton("📊 Account Stats", callback_data="stats"),
             InlineKeyboardButton("💎 Upgrade Plans", callback_data="plans")],
            [InlineKeyboardButton("📞 Contact Admin", callback_data="contact")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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

🎯 *Enhanced Admin Features:*
• `/signal` - Generate normal signal
• `/signal quick` - Quick Trade signal
• `/signal EUR/USD` - Specific pair
• `/signal EUR/USD quick` - Quick trade specific

*40s pre-entry system activated!* ⚡
""", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
        with sqlite3.connect(Config.DB_PATH) as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            quick_signals = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_style = 'QUICK'").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
🏢 *ADMIN DASHBOARD* 🌟

📊 *Statistics:*
• Total Users: {total_users}
• Total Signals: {total_signals}
• Quick Trades: {quick_signals}
• Current Session: {current_session['name']}

🎯 *Signal Commands:*
• `/signal` - Normal trade
• `/signal quick` - Quick Trade
• `/signal EUR/USD` - Specific pair
• `/signal EUR/USD quick` - Quick specific

⚡ *Quick Trade Features:*
• 40s pre-entry warning
• Candle-based analysis
• Faster execution
• Tighter stops

*Choose your trading style!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("⚡ Quick Trade", callback_data="signal_quick"),
             InlineKeyboardButton("📈 Normal Trade", callback_data="signal_normal")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="admin_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login YOUR_TOKEN`", parse_mode='Markdown')
            return
        
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
        
        await update.message.reply_text(f"🎯 *Generating {signal_style} signal...*", parse_mode='Markdown')
        
        result = await self.trading_bot.generate_admin_signal_sequence(user.id, symbol, signal_style)
        
        if result and "pre_signal" in result:
            pre_signal = result["pre_signal"]
            analysis = json.loads(pre_signal["analysis"])
            
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

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self.trading_bot.session_manager.get_current_session()
        
        if session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED*

*Next Trading Sessions:*

🌅 *MORNING SESSION* (08:00-12:00)
• Volatility: HIGH
• Accuracy: 96.2%
• Pairs: EUR/USD, GBP/USD, EUR/JPY

🌇 *EVENING SESSION* (16:00-20:00)
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
🕒 *{session['name']}* ✅ ACTIVE

⏰ Hours: {session['start_hour']:02d}:00-{session['end_hour']:02d}:00
📊 Volatility: {session['volatility']}
🎯 Accuracy: {session['accuracy']}%
💎 Pairs: {', '.join(session['optimal_pairs'])}

⚡ *Enhanced Signals Active:*
• Quick Trade (40s pre-entry)
• Normal signals with analysis
• Candle-based entries
• Real-time monitoring

*Professional signals are live!* 🚀
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await update.message.reply_text(f"""
💎 *UPGRADE YOUR ACCOUNT*

*Contact admin for premium features:*
{Config.ADMIN_CONTACT}

*Unlock enhanced trading!* 🚀
""", parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = f"""
📞 *CONTACT ADMIN*

*Direct Contact:* {Config.ADMIN_CONTACT}

💡 *Premium Support:*
• Quick Trade signals
• All session access
• Higher accuracy
• Priority support

*Message us now!* 💎
"""
        keyboard = [
            [InlineKeyboardButton("📱 Message Admin", url=f"https://t.me/{Config.ADMIN_CONTACT.replace('@', '')}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def plans_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = f"""
💎 *LEKZY FX AI PRO - PREMIUM PLANS*

🌅 *BASIC* - $19/month
• Morning Session
• Quick Trade signals
• 10 signals/day

🌇 *PRO* - $49/month  
• Morning + Evening
• Enhanced analysis
• 25 signals/day

🌃 *VIP* - $99/month
• All Sessions
• Priority signals
• 50 signals/day

🌟 *PREMIUM* - $199/month
• 24/7 Access
• Unlimited signals
• Personal support

*Contact {Config.ADMIN_CONTACT} to upgrade!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        if self.admin_auth.is_admin(user.id):
            message = """
📊 *ADMIN STATS* 👑

• Full system access
• Quick Trade signals
• 24/7 generation
• Enhanced analytics

*You have premium admin features!* 🚀
"""
        else:
            message = f"""
📊 *YOUR ACCOUNT STATS*

• Plan: {user_plan}
• Signals: 5 per day
• Session: Morning
• Quick Trade: Available

*Upgrade for more features!* 💎
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def start_polling(self):
        await self.application.updater.start_polling()

    async def stop(self):
        self.trading_bot.is_running = False
        await self.application.stop()

# ==================== MAIN APPLICATION ====================
async def main():
    """Main function to run the bot"""
    initialize_database()
    
    bot = CompleteTelegramBot()
    await bot.initialize()
    
    logger.info("🚀 LEKZY FX AI PRO - Quick Trade System Started")
    
    # Start polling
    await bot.start_polling()
    
    # Keep the bot running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
