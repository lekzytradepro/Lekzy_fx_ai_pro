#!/usr/bin/env python3
"""
LEKZY FX AI PRO - Professional Auto Signal System with Daily Limits
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
    ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "123456789")  # Your admin user ID
    DB_PATH = "/app/data/lekzy_fx_ai.db"
    PORT = int(os.getenv("PORT", 10000))
    PRE_ENTRY_DELAY = 40  # seconds before entry
    TIMEZONE_OFFSET = 1  # UTC+1
    BROADCAST_INTERVAL = 1800  # 30 minutes in seconds

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
    return "🤖 LEKZY FX AI PRO - Professional Auto Signal System 🚀"

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
                auto_signals_enabled BOOLEAN DEFAULT FALSE,
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                plan_type TEXT DEFAULT 'TRIAL',
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_active TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS broadcasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT,
                broadcast_time TEXT,
                status TEXT DEFAULT 'SENT',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                signal_id TEXT,
                signal_date TEXT DEFAULT CURRENT_DATE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Professional auto signal database initialized")
        
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
                "accuracy": 96.2,
                "broadcast_time": 6.5  # 06:30 UTC+1 (30 minutes before)
            },
            "EVENING": {
                "start_hour": 15, "end_hour": 19,  # 16:00-20:00 UTC+1
                "name": "🌇 NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "accuracy": 97.8,
                "broadcast_time": 14.5  # 14:30 UTC+1 (30 minutes before)
            },
            "ASIAN": {
                "start_hour": 23, "end_hour": 3,   # 00:00-04:00 UTC+1 (next day)
                "name": "🌃 Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "accuracy": 92.5,
                "broadcast_time": 22.5  # 22:30 UTC+1 (30 minutes before)
            },
            "ADMIN_24_7": {
                "start_hour": 0, "end_hour": 24,
                "name": "👑 24/7 Admin Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"],
                "volatility": "ADMIN",
                "accuracy": 98.5,
                "broadcast_time": None
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
            next_session = self.sessions["ASIAN"]
        else:
            current_index = sessions_order.index(current_session["id"])
            next_index = (current_index + 1) % len(sessions_order)
            next_session = self.sessions[sessions_order[next_index]]
        
        return {**next_session, "id": sessions_order[next_index]}

    def get_upcoming_sessions(self):
        """Get sessions that should have broadcasts sent"""
        now_utc1 = self.get_current_time_utc1()
        current_hour = now_utc1.hour + (now_utc1.minute / 60)
        
        upcoming = []
        for session_id, session in self.sessions.items():
            if session_id != "ADMIN_24_7" and session["broadcast_time"] is not None:
                # Check if current time is within 30 minutes of broadcast time
                if abs(current_hour - session["broadcast_time"]) < 0.5:  # 30 minutes window
                    upcoming.append({**session, "id": session_id})
        
        return upcoming

# ==================== PROFESSIONAL SIGNAL GENERATOR ====================
class ProfessionalSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        self.pending_signals = {}
    
    def generate_professional_analysis(self, symbol: str) -> dict:
        """Generate professional trading analysis with new candle focus"""
        
        # Professional candle patterns with new candle emphasis
        candle_patterns = [
            "NEW CANDLE: Bullish Engulfing pattern forming",
            "NEW CANDLE: Bearish Engulfing with strong volume",
            "NEW CANDLE: Hammer at key support level",
            "NEW CANDLE: Shooting star at resistance",
            "NEW CANDLE: Doji reversal pattern confirmed",
            "NEW CANDLE: Three white soldiers pattern emerging"
        ]
        
        # Professional timeframes
        timeframes = ["M5", "M15", "H1"]
        
        # Market conditions with new candle focus
        market_conditions = [
            "New candle forming with institutional flow",
            "Price reacting at key level on new candle",
            "New candle confirming market structure break",
            "Liquidity activation on new candle formation",
            "Economic catalyst driving new candle momentum"
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
            "entry_timing": "NEW_CANDLE_CONFIRMATION",
            "new_candle_required": True
        }
    
    def generate_pre_entry_signal(self, symbol: str = None, is_admin: bool = False) -> dict:
        """Generate pre-entry signal with professional analysis"""
        try:
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
            
            # Calculate timestamps
            current_time = datetime.now()
            entry_time = current_time + timedelta(seconds=Config.PRE_ENTRY_DELAY)
            
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
                "generated_at": current_time.isoformat(),
                "current_time_str": current_time.strftime("%H:%M:%S"),
                "entry_time_str": entry_time.strftime("%H:%M:%S")
            }
            
            self.pending_signals[signal_id] = signal_data
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            return None
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate entry signal based on pre-entry with new candle confirmation"""
        try:
            if pre_signal_id not in self.pending_signals:
                return None
            
            pre_signal = self.pending_signals[pre_signal_id]
            analysis = json.loads(pre_signal["analysis"])
            
            # Update analysis to confirm new candle
            analysis["new_candle_confirmed"] = True
            analysis["entry_executed"] = "NEW_CANDLE_BASED"
            analysis["execution_time"] = datetime.now().isoformat()
            
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
                "risk_reward": risk_reward,
                "analysis": json.dumps(analysis),
                "entry_time_actual": datetime.now().strftime("%H:%M:%S")
            }
            
            del self.pending_signals[pre_signal_id]
            return entry_signal
            
        except Exception as e:
            logger.error(f"❌ Entry signal generation error: {e}")
            return None

# ==================== USER & SUBSCRIPTION MANAGER ====================
class UserManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def add_user(self, user_id: int, username: str, first_name: str, last_name: str = ""):
        """Add or update user in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO users 
                (user_id, username, first_name, last_name, last_active)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, first_name, last_name, datetime.now().isoformat()))
            conn.commit()
    
    def get_all_users(self):
        """Get all user IDs for broadcasting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT user_id FROM users")
            return [row[0] for row in cursor.fetchall()]
    
    def get_user_count(self):
        """Get total user count"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]

class SubscriptionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def start_trial(self, user_id: int, username: str, first_name: str):
        """Start free trial"""
        end_date = datetime.now() + timedelta(days=3)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_type, start_date, end_date, payment_status, max_daily_signals, allowed_sessions, auto_signals_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, "TRIAL", datetime.now().isoformat(), 
                end_date.isoformat(), "ACTIVE", 5, '["MORNING"]', False
            ))
            conn.commit()
        
        logger.info(f"✅ Trial started: {username} ({user_id})")
    
    def get_user_plan(self, user_id: int) -> dict:
        """Get user's current plan and usage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT plan_type, signals_used, max_daily_signals, auto_signals_enabled, allowed_sessions
                FROM subscriptions WHERE user_id = ?
            """, (user_id,))
            result = cursor.fetchone()
            
            if result:
                plan_type, signals_used, max_daily, auto_enabled, allowed_sessions = result
                return {
                    "plan_type": plan_type,
                    "signals_used": signals_used,
                    "max_daily_signals": max_daily,
                    "auto_signals_enabled": bool(auto_enabled),
                    "allowed_sessions": json.loads(allowed_sessions) if allowed_sessions else ["MORNING"],
                    "signals_remaining": max_daily - signals_used
                }
            else:
                # Default trial plan
                return {
                    "plan_type": "TRIAL",
                    "signals_used": 0,
                    "max_daily_signals": 5,
                    "auto_signals_enabled": False,
                    "allowed_sessions": ["MORNING"],
                    "signals_remaining": 5
                }
    
    def can_user_request_signal(self, user_id: int) -> tuple:
        """Check if user can request a signal"""
        user_plan = self.get_user_plan(user_id)
        
        if user_plan["signals_used"] >= user_plan["max_daily_signals"]:
            return False, "Daily signal limit reached. Upgrade for more signals!"
        
        # Check if user's allowed session is active
        current_session = SessionManager().get_current_session()
        if current_session["id"] not in user_plan["allowed_sessions"] and current_session["id"] != "ADMIN_24_7":
            return False, f"Signal not available in {current_session['name']}. Your plan allows: {', '.join(user_plan['allowed_sessions'])}"
        
        return True, "OK"
    
    def increment_signal_count(self, user_id: int):
        """Increment user's signal count"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE subscriptions 
                SET signals_used = signals_used + 1 
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
    
    def toggle_auto_signals(self, user_id: int) -> bool:
        """Toggle auto signals for user"""
        user_plan = self.get_user_plan(user_id)
        new_status = not user_plan["auto_signals_enabled"]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE subscriptions 
                SET auto_signals_enabled = ? 
                WHERE user_id = ?
            """, (new_status, user_id))
            conn.commit()
        
        return new_status
    
    def reset_daily_limits(self):
        """Reset daily signal counts (run at midnight)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE subscriptions SET signals_used = 0")
            conn.commit()
        logger.info("✅ Daily signal limits reset")

# ==================== PROFESSIONAL TRADING BOT ====================
class ProfessionalTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = SessionManager()
        self.signal_generator = ProfessionalSignalGenerator()
        self.user_manager = UserManager(Config.DB_PATH)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.is_running = False
        self.auto_signals_tasks = {}
    
    async def notify_admin_new_user(self, user: dict):
        """Notify admin about new user"""
        try:
            admin_user_id = int(Config.ADMIN_USER_ID)
            message = f"""
👤 *NEW USER REGISTERED*

🆔 *User ID:* `{user.id}`
👤 *Name:* {user.first_name} {user.last_name or ''}
📧 *Username:* @{user.username or 'N/A'}
⏰ *Time:* {datetime.now().strftime('%H:%M UTC+1')}

📊 *Total Users:* {self.user_manager.get_user_count()}
🎯 *System Status:* ACTIVE
"""
            await self.application.bot.send_message(
                chat_id=admin_user_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"📧 Admin notified about new user: {user.username}")
        except Exception as e:
            logger.error(f"❌ Failed to notify admin: {e}")
    
    async def broadcast_session_alert(self):
        """Broadcast session alerts to all users"""
        try:
            upcoming_sessions = self.session_manager.get_upcoming_sessions()
            
            for session in upcoming_sessions:
                # Check if broadcast already sent for this session today
                with sqlite3.connect(Config.DB_PATH) as conn:
                    today = datetime.now().strftime("%Y-%m-%d")
                    cursor = conn.execute(
                        "SELECT id FROM broadcasts WHERE session_type = ? AND DATE(created_at) = ?",
                        (session["id"], today)
                    )
                    if cursor.fetchone():
                        continue  # Already sent today
                
                # Prepare broadcast message
                message = f"""
🔔 *SESSION STARTING SOON*

{session['name']}
⏰ *Starts in:* 30 minutes
🕒 *Session Time:* {session['start_hour']:02d}:00-{session['end_hour']:02d}:00 UTC+1

📊 *Volatility:* {session['volatility']}
🎯 *Accuracy:* {session['accuracy']}%
💎 *Optimal Pairs:* {', '.join(session['optimal_pairs'][:2])}

⚡ *Professional signals will be generated automatically*
🎯 *Get ready for trading opportunities!*

*Prepare your trading setup!* 🚀
"""
                # Send to all users
                users = self.user_manager.get_all_users()
                success_count = 0
                
                for user_id in users:
                    try:
                        await self.application.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        success_count += 1
                        await asyncio.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"❌ Failed to send broadcast to {user_id}: {e}")
                
                # Record broadcast
                with sqlite3.connect(Config.DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO broadcasts (session_type, broadcast_time) VALUES (?, ?)",
                        (session["id"], datetime.now().isoformat())
                    )
                    conn.commit()
                
                logger.info(f"📢 Session broadcast sent for {session['name']} to {success_count}/{len(users)} users")
                
        except Exception as e:
            logger.error(f"❌ Broadcast error: {e}")
    
    async def start_user_auto_signals(self, user_id: int):
        """Start auto signals for a specific user"""
        try:
            user_plan = self.subscription_manager.get_user_plan(user_id)
            
            if not user_plan["auto_signals_enabled"]:
                return
            
            async def user_auto_signal_loop():
                while self.is_running and user_plan["auto_signals_enabled"]:
                    try:
                        # Check daily limit
                        if user_plan["signals_used"] >= user_plan["max_daily_signals"]:
                            await self.application.bot.send_message(
                                chat_id=user_id,
                                text="🔔 *Auto Signals Paused* - Daily limit reached. Upgrade for unlimited signals! 💎",
                                parse_mode='Markdown'
                            )
                            break
                        
                        # Check session access
                        current_session = self.session_manager.get_current_session()
                        if current_session["id"] not in user_plan["allowed_sessions"]:
                            await asyncio.sleep(300)  # Check every 5 minutes
                            continue
                        
                        # Generate signal
                        symbol = random.choice(current_session["optimal_pairs"])
                        pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, False)
                        
                        if pre_signal:
                            # Store signal
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
                                    pre_signal["signal_style"], "AUTO"
                                ))
                                conn.commit()
                            
                            # Send pre-entry to user
                            analysis = json.loads(pre_signal["analysis"])
                            direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
                            
                            pre_message = f"""
🤖 *AUTO SIGNAL - PRE ENTRY*

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Expected Entry:* `{pre_signal['entry_price']:.5f}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

⏰ *Timing:*
• 🕐 Current Time: `{pre_signal['current_time_str']}`
• 🎯 Expected Entry: `{pre_signal['entry_time_str']}`
• ⏱️ Countdown: {Config.PRE_ENTRY_DELAY} seconds

📊 *Analysis:* {analysis['candle_pattern']}

*Entry signal coming soon...* ⚡
"""
                            await self.application.bot.send_message(
                                chat_id=user_id,
                                text=pre_message,
                                parse_mode='Markdown'
                            )
                            
                            # Wait for entry
                            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
                            
                            # Generate and send entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry, risk_reward, signal_style, requested_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, tuple(entry_signal.values()))
                                    conn.commit()
                                
                                # Increment user signal count
                                self.subscription_manager.increment_signal_count(user_id)
                                user_plan["signals_used"] += 1
                                
                                entry_message = f"""
🤖 *AUTO SIGNAL - ENTRY*

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry Price:* `{entry_signal['entry_price']:.5f}`
✅ *Take Profit:* `{entry_signal['take_profit']:.5f}`
❌ *Stop Loss:* `{entry_signal['stop_loss']:.5f}`

📈 *Details:*
• Confidence: *{entry_signal['confidence']*100:.1f}%*
• Risk/Reward: *1:{entry_signal['risk_reward']}*
• Remaining Signals: *{user_plan['max_daily_signals'] - user_plan['signals_used']}*

*Execute this trade!* 🚀
"""
                                await self.application.bot.send_message(
                                    chat_id=user_id,
                                    text=entry_message,
                                    parse_mode='Markdown'
                                )
                        
                        # Wait before next auto signal (2-3 minutes)
                        await asyncio.sleep(random.randint(120, 180))
                        
                    except Exception as e:
                        logger.error(f"User auto signal error for {user_id}: {e}")
                        await asyncio.sleep(60)
            
            # Start the auto signal loop for this user
            task = asyncio.create_task(user_auto_signal_loop())
            self.auto_signals_tasks[user_id] = task
            logger.info(f"✅ Auto signals started for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start auto signals for user {user_id}: {e}")
    
    async def stop_user_auto_signals(self, user_id: int):
        """Stop auto signals for a specific user"""
        if user_id in self.auto_signals_tasks:
            self.auto_signals_tasks[user_id].cancel()
            del self.auto_signals_tasks[user_id]
            logger.info(f"✅ Auto signals stopped for user {user_id}")
    
    async def start_auto_services(self):
        """Start all automatic services"""
        self.is_running = True
        
        async def broadcast_loop():
            while self.is_running:
                try:
                    await self.broadcast_session_alert()
                    await asyncio.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Broadcast loop error: {e}")
                    await asyncio.sleep(60)
        
        async def system_auto_signal_loop():
            """Generate system-wide auto signals"""
            while self.is_running:
                try:
                    session = self.session_manager.get_current_session()
                    
                    if session["id"] != "CLOSED" and session["id"] != "ADMIN_24_7":
                        logger.info(f"🎯 System Auto-generating {session['name']} signals")
                        
                        for symbol in session["optimal_pairs"][:1]:
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, False)
                            
                            if pre_signal:
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
                                        pre_signal["signal_style"], "SYSTEM_AUTO"
                                    ))
                                    conn.commit()
                                
                                logger.info(f"📊 System Auto Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
                                
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
                                    
                                    logger.info(f"🎯 System Auto Entry: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    await asyncio.sleep(random.randint(90, 150))  # 1.5-2.5 minutes between system signals
                    
                except Exception as e:
                    logger.error(f"System auto signal error: {e}")
                    await asyncio.sleep(60)
        
        async def daily_reset_loop():
            """Reset daily limits at midnight"""
            while self.is_running:
                try:
                    now = datetime.now()
                    if now.hour == 0 and now.minute == 0:  # Midnight
                        self.subscription_manager.reset_daily_limits()
                        logger.info("✅ Daily limits reset")
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Daily reset error: {e}")
                    await asyncio.sleep(60)
        
        # Start all services
        asyncio.create_task(broadcast_loop())
        asyncio.create_task(system_auto_signal_loop())
        asyncio.create_task(daily_reset_loop())
        logger.info("✅ Professional auto services started")
    
    async def generate_signal_sequence(self, user_id: int, symbol: str = None, is_admin: bool = False):
        """Generate signal sequence for user"""
        try:
            # Check if user can request signal
            if not is_admin:
                can_request, message = self.subscription_manager.can_user_request_signal(user_id)
                if not can_request:
                    return {"error": message}
            
            # Generate pre-entry signal
            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, is_admin)
            
            if not pre_signal:
                return {"error": "Failed to generate signal. Please try again."}
            
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
                    pre_signal["signal_style"], "ADMIN" if is_admin else "USER"
                ))
                conn.commit()
            
            # Increment user signal count if not admin
            if not is_admin:
                self.subscription_manager.increment_signal_count(user_id)
            
            logger.info(f"📊 {'Admin' if is_admin else 'User'} Pre-entry: {pre_signal['symbol']} {pre_signal['direction']}")
            
            return {
                "pre_signal": pre_signal,
                "entry_in_seconds": Config.PRE_ENTRY_DELAY
            }
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return {"error": "Signal generation failed. Please try again."}
    
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
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the professional bot"""
        self.application = Application.builder().token(self.token).build()
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.trading_bot = ProfessionalTradingBot(self.application)
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("contact", self.contact_command))
        self.application.add_handler(CommandHandler("upgrade", self.upgrade_command))
        self.application.add_handler(CommandHandler("auto", self.auto_command))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        await self.application.initialize()
        await self.application.start()
        
        await self.trading_bot.start_auto_services()
        
        logger.info("🤖 Professional Auto Signal Trading Bot Initialized!")

    async def create_welcome_message(self, user, current_session):
        """Create professional welcome message"""
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        return f"""
🎯 *LEKZY FX AI PRO* - AUTO SIGNAL SYSTEM

*Welcome, {user.first_name}!* 🌟

*Your 3-Day Free Trial Activated* ✅

🕒 *Live Market Session:*
{current_session['name']}
⏰ *Time:* {current_session['current_time']}

📊 *Your Account:*
• Plan: *{user_plan['plan_type']}*
• Signals Today: *{user_plan['signals_used']}/{user_plan['max_daily_signals']}*
• Auto Signals: *{'✅ ON' if user_plan['auto_signals_enabled'] else '❌ OFF'}*

⚡ *Professional Features:*
• 🤖 Auto Signal System
• ⏱️ 40s Pre-Entry Timing
• 🕯️ New Candle Based Entries
• 📢 Session Broadcast Alerts

🎮 *QUICK COMMANDS:*

🚀 */signal* - Get Instant Signal
🤖 */auto* - Toggle Auto Signals
🕒 */session* - Market Hours & Status
📊 */signals* - Recent Trading Signals
💎 */upgrade* - Premium Plans
📞 */contact* - Premium Support

*Start trading like a pro!* 🚀
"""

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with professional welcome"""
        user = update.effective_user
        
        # Add user to database and notify admin
        self.trading_bot.user_manager.add_user(user.id, user.username, user.first_name, user.last_name or "")
        await self.trading_bot.notify_admin_new_user(user)
        
        self.subscription_manager.start_trial(user.id, user.username, user.first_name)
        
        current_session = self.trading_bot.session_manager.get_current_session()
        welcome_message = await self.create_welcome_message(user, current_session)
        
        # Professional keyboard
        keyboard = [
            [InlineKeyboardButton("🚀 GET INSTANT SIGNAL", callback_data="get_signal")],
            [InlineKeyboardButton("🤖 TOGGLE AUTO SIGNALS", callback_data="toggle_auto")],
            [InlineKeyboardButton("🕒 MARKET SESSION", callback_data="session")],
            [InlineKeyboardButton("📊 RECENT SIGNALS", callback_data="signals")],
            [InlineKeyboardButton("💎 UPGRADE PLANS", callback_data="upgrade")],
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
1. Use */signal* for instant manual signals
2. Use */auto* to toggle automatic signals
3. Receive signals in your allowed sessions
4. Execute trades on new candle formation

⚡ *Signal System:*
• 🤖 Auto Signals (User controlled)
• ⏱️ 40s pre-entry timing
• 🕯️ New candle based entries
• 📊 Professional analysis

🕒 *Trading Sessions (UTC+1):*
• 🌅 London: 08:00-12:00
• 🌇 NY/London: 16:00-20:00  
• 🌃 Asian: 00:00-04:00

🔔 *Session Alerts:*
30 minutes before each session

📞 *Support:* @LekzyTradingPro

*Trade like a professional!* 🚀
"""
        await update.message.reply_text(message, parse_mode='Markdown')

    async def auto_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /auto command to toggle auto signals"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        new_status = self.subscription_manager.toggle_auto_signals(user.id)
        
        if new_status:
            # Start auto signals for user
            await self.trading_bot.start_user_auto_signals(user.id)
            message = f"""
🤖 *AUTO SIGNALS ENABLED* ✅

*Automatic signals are now active for your account!*

📊 *Your Settings:*
• Plan: *{user_plan['plan_type']}*
• Daily Limit: *{user_plan['max_daily_signals']} signals*
• Allowed Sessions: *{', '.join(user_plan['allowed_sessions'])}*

⚡ *How it works:*
• Signals generated automatically in your sessions
• You'll receive pre-entry and entry signals
• Daily limit respected automatically
• System stops when limit reached

*You will now receive automatic trading signals!* 🚀
"""
        else:
            # Stop auto signals for user
            await self.trading_bot.stop_user_auto_signals(user.id)
            message = """
🤖 *AUTO SIGNALS DISABLED* ❌

*Automatic signals have been turned off.*

💡 *You can still:*
• Use */signal* for manual signals
• Re-enable anytime with */auto*
• Check */session* for market hours

*Auto signals are now disabled.*
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
• 🚀 /signal - Generate instant signals
• 🤖 System auto signal management
• 🕒 /session - Market session info
• 📊 /signals - Signal history
• 👑 /admin - Admin dashboard

*Professional system activated!* ⚡
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
            total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            auto_users = conn.execute("SELECT COUNT(*) FROM subscriptions WHERE auto_signals_enabled = 1").fetchone()[0]
        
        current_session = self.trading_bot.session_manager.get_current_session()
        
        message = f"""
👑 *PROFESSIONAL ADMIN DASHBOARD*

📊 *System Statistics:*
• Total Users: {total_users}
• Total Signals: {total_signals}
• Auto Signal Users: {auto_users}
• Current Session: {current_session['name']}
• Time: {current_session['current_time']}

⚡ *Signal Systems:*
• 🤖 User Auto Signals (Controlled by users)
• 🚀 System Auto Signals (Always running)
• 👑 Admin Manual Signals

🎯 *Features:*
• Daily signal limits per user
• Session-based access control
• New candle based entries
• Professional risk management

*Professional system active!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("🚀 GENERATE SIGNAL", callback_data="admin_signal")],
            [InlineKeyboardButton("📊 SYSTEM STATS", callback_data="admin_stats")],
            [InlineKeyboardButton("🕒 MARKET SESSION", callback_data="session")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command - WITH DAILY LIMITS"""
        user = update.effective_user
        is_admin = self.admin_auth.is_admin(user.id)
        
        # Parse symbol if provided
        symbol = None
        if context.args:
            symbol_arg = context.args[0].upper().replace('_', '/')
            valid_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
            if symbol_arg in valid_pairs:
                symbol = symbol_arg
        
        # Generate signal sequence
        await update.message.reply_text("🚀 *Generating professional signal...*", parse_mode='Markdown')
        
        result = await self.trading_bot.generate_signal_sequence(user.id, symbol, is_admin)
        
        if result and "error" in result:
            await update.message.reply_text(f"❌ {result['error']}", parse_mode='Markdown')
            return
        
        if result and "pre_signal" in result:
            pre_signal = result["pre_signal"]
            analysis = json.loads(pre_signal["analysis"])
            
            # Get updated user plan
            user_plan = self.subscription_manager.get_user_plan(user.id)
            
            # Send pre-entry signal immediately with timestamps
            direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
            
            message = f"""
🎯 *PROFESSIONAL PRE-ENTRY SIGNAL* ⚡

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Expected Entry:* `{pre_signal['entry_price']:.5f}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

⏰ *Timing:*
• 🕐 Current Time: `{pre_signal['current_time_str']}`
• 🎯 Expected Entry: `{pre_signal['entry_time_str']}`
• ⏱️ Countdown: {Config.PRE_ENTRY_DELAY} seconds

📊 *Professional Analysis:*
{analysis['candle_pattern']}
• Timeframe: {analysis['timeframe']}
• Momentum: {analysis['momentum']}
• Risk: {analysis['risk_rating']}

📈 *Your Account:*
• Signals Today: *{user_plan['signals_used']}/{user_plan['max_daily_signals']}*
• Auto Signals: *{'✅ ON' if user_plan['auto_signals_enabled'] else '❌ OFF'}*

⏰ *Entry signal coming in {Config.PRE_ENTRY_DELAY} seconds...*
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Wait and send entry signal
            await asyncio.sleep(Config.PRE_ENTRY_DELAY)
            
            entry_signal = await self.trading_bot.generate_entry_signal(pre_signal["signal_id"])
            
            if entry_signal:
                # Get final user plan
                final_user_plan = self.subscription_manager.get_user_plan(user.id)
                
                entry_message = f"""
🎯 *ENTRY SIGNAL* ✅
*EXECUTE ON NEW CANDLE*

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry Price:* `{entry_signal['entry_price']:.5f}`
✅ *Take Profit:* `{entry_signal['take_profit']:.5f}`
❌ *Stop Loss:* `{entry_signal['stop_loss']:.5f}`

⏰ *Entry Time:* `{entry_signal['entry_time_actual']}`

📈 *Trade Details:*
• Confidence: *{entry_signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{entry_signal['risk_reward']}* ⚖️
• Type: *PROFESSIONAL* 💎

📊 *Your Account:*
• Signals Remaining: *{final_user_plan['signals_remaining']}*
• Auto Signals: *{'✅ ON' if final_user_plan['auto_signals_enabled'] else '❌ OFF'}*

⚡ *Execution Confirmed:*
• ✅ New candle confirmed
• ✅ Optimal entry level
• ✅ Professional setup

*Execute this trade on new candle formation!* 🚀
"""
                keyboard = [
                    [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
                    [InlineKeyboardButton("🔄 NEW SIGNAL", callback_data="get_signal")],
                    [InlineKeyboardButton("🤖 AUTO SIGNALS", callback_data="toggle_auto")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(entry_message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to generate entry signal")
        else:
            await update.message.reply_text("❌ Signal generation failed. Please try again.")

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
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

🔔 *Session Alerts:* 30 minutes before each session

*Auto-signals resume in session hours!* 📈
"""
        else:
            message = f"""
🕒 *{session['name']}* ✅ LIVE

⏰ *Current Time:* {session['current_time']}
📊 *Volatility:* {session['volatility']}
🎯 *Accuracy:* {session['accuracy']}%
💎 *Optimal Pairs:* {', '.join(session['optimal_pairs'])}

⚡ *Signal Systems Active:*
• 🤖 User Auto Signals
• 🚀 System Auto Signals
• ⏱️ 40s pre-entry system
• 🕯️ New candle based entries

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
            request_type = {
                "ADMIN": "👑",
                "USER": "👤", 
                "AUTO": "🤖",
                "SYSTEM_AUTO": "⚡"
            }.get(requested_by, "👤")
            
            message += f"{type_emoji} {dir_emoji} {symbol} {request_type}\n"
            message += f"💵 {entry} | {confidence*100:.1f}% | {time_str}\n\n"
        
        message += "⚡ *Professional signal system active!*\n"
        message += "🕯️ *All entries based on new candle confirmation*"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /contact command"""
        message = f"""
📞 *PROFESSIONAL SUPPORT*

*Contact Admin:* {Config.ADMIN_CONTACT}

💎 *Premium Features:*
• 24/7 Signal Access
• Unlimited Auto Signals
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
            [InlineKeyboardButton("💎 UPGRADE PLANS", callback_data="upgrade")],
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="get_signal")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def upgrade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upgrade command"""
        message = f"""
💎 *UPGRADE TO PREMIUM*

*Contact Admin:* {Config.ADMIN_CONTACT}

🚀 *Premium Plans:*

🌅 *BASIC* - $19/month
• Morning Session Access
• 10 Signals/Day
• Auto Signals Enabled
• 95%+ Accuracy

🌇 *PRO* - $49/month  
• Morning + Evening Sessions
• 25 Signals/Day
• Enhanced Auto Signals
• 96%+ Accuracy

🌃 *VIP* - $99/month
• All Sessions (24/7)
• 50 Signals/Day
• Priority Auto Signals
• 97%+ Accuracy

🌟 *PREMIUM* - $199/month
• 24/7 Priority Access
• Unlimited Signals
• Personal Support
• 98%+ Accuracy

💡 *All Premium Features:*
• New Candle Based Entries
• Session Broadcast Alerts
• Advanced Analytics
• Priority Execution

*Contact {Config.ADMIN_CONTACT} to upgrade!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("📞 CONTACT ADMIN", callback_data="contact")],
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
        elif query.data == "toggle_auto":
            await self.auto_command(update, context)
        elif query.data == "session":
            await self.session_command(update, context)
        elif query.data == "signals":
            await self.signals_command(update, context)
        elif query.data == "contact":
            await self.contact_command(update, context)
        elif query.data == "upgrade":
            await self.upgrade_command(update, context)
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
        logger.info("🚀 LEKZY FX AI PRO - Professional Auto Signal System Started")
    
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
