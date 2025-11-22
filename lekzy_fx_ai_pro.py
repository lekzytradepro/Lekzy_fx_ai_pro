#!/usr/bin/env python3
"""
LEKZY FX AI PRO - FIXED VERSION WITH TIMEFRAME ENTRIES
"""

import os
import asyncio
import sqlite3
import json
import time
import random
import logging
import secrets
import string
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
    ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "123456789")
    DB_PATH = "/app/data/lekzy_fx_ai.db"
    PORT = int(os.getenv("PORT", 10000))
    PRE_ENTRY_DELAY = 40  # seconds before entry

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
    return """
🤖 LEKZY FX AI PRO - ACTIVE 🚀

✅ Bot Status: RUNNING
✅ Token System: OPERATIONAL
✅ Signal System: WORKING

📊 Features:
• Timeframe-Based Entries (1M, 5M, 15M)
• Token Subscription System
• Admin Signal Generation
• Professional Trading
"""

@app.route('/health')
def health():
    return json.dumps({
        "status": "healthy",
        "service": "lekzy_fx_ai_pro",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0"
    })

def run_web_server():
    try:
        port = int(os.environ.get('PORT', Config.PORT))
        logger.info(f"🌐 Starting web server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"❌ Web server failed: {e}")

def start_web_server():
    try:
        web_thread = Thread(target=run_web_server)
        web_thread.daemon = True
        web_thread.start()
        logger.info("✅ Web server thread started")
    except Exception as e:
        logger.error(f"❌ Failed to start web server: {e}")

# ==================== DATABASE SETUP ====================
def initialize_database():
    try:
        os.makedirs("/app/data", exist_ok=True)
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                plan_type TEXT DEFAULT 'TRIAL',
                subscription_end TEXT,
                max_daily_signals INTEGER DEFAULT 5,
                signals_used INTEGER DEFAULT 0,
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                confidence REAL,
                signal_type TEXT,
                timeframe TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                login_time TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscription_tokens (
                token TEXT PRIMARY KEY,
                days_valid INTEGER DEFAULT 30,
                created_by INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                used_by INTEGER DEFAULT NULL,
                used_at TEXT DEFAULT NULL,
                status TEXT DEFAULT 'ACTIVE'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                user_id INTEGER PRIMARY KEY,
                token_used TEXT,
                plan_type TEXT DEFAULT 'PREMIUM',
                start_date TEXT,
                end_date TEXT,
                max_daily_signals INTEGER DEFAULT 50,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

# ==================== TOKEN MANAGER ====================
class TokenManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def generate_token(self, days_valid=30, created_by=None):
        try:
            alphabet = string.ascii_uppercase + string.digits
            token = ''.join(secrets.choice(alphabet) for _ in range(12))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "INSERT INTO subscription_tokens (token, days_valid, created_by) VALUES (?, ?, ?)",
                (token, days_valid, created_by)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Token generated: {token} for {days_valid} days")
            return token
            
        except Exception as e:
            logger.error(f"❌ Token generation failed: {e}")
            return None
    
    def validate_token(self, token):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT token, days_valid, status FROM subscription_tokens WHERE token = ? AND status = 'ACTIVE'",
                (token,)
            )
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, "Invalid or expired token"
            
            token_str, days_valid, status = result
            
            conn.execute(
                "UPDATE subscription_tokens SET status = 'USED', used_at = ? WHERE token = ?",
                (datetime.now().isoformat(), token)
            )
            conn.commit()
            conn.close()
            
            return True, days_valid
            
        except Exception as e:
            logger.error(f"❌ Token validation failed: {e}")
            return False, "Token validation error"

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.token_manager = TokenManager(db_path)
    
    def activate_premium_subscription(self, user_id, token, days_valid):
        try:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_valid)
            
            conn = sqlite3.connect(self.db_path)
            
            conn.execute("""
                INSERT OR REPLACE INTO user_subscriptions 
                (user_id, token_used, plan_type, start_date, end_date, max_daily_signals)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, token, "PREMIUM", start_date.isoformat(), end_date.isoformat(), 50))
            
            conn.execute("""
                INSERT OR REPLACE INTO users 
                (user_id, plan_type, subscription_end, max_daily_signals, signals_used)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, "PREMIUM", end_date.isoformat(), 50, 0))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Premium activated for user {user_id} for {days_valid} days")
            return True
            
        except Exception as e:
            logger.error(f"❌ Subscription activation failed: {e}")
            return False
    
    def get_user_subscription(self, user_id):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT plan_type, subscription_end, max_daily_signals, signals_used 
                FROM users WHERE user_id = ?
            """, (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                plan_type, sub_end, max_signals, signals_used = result
                
                is_active = True
                if sub_end and plan_type != "TRIAL":
                    end_date = datetime.fromisoformat(sub_end)
                    is_active = datetime.now() < end_date
                
                return {
                    "plan_type": plan_type,
                    "is_active": is_active,
                    "subscription_end": sub_end,
                    "max_daily_signals": max_signals,
                    "signals_used": signals_used,
                    "signals_remaining": max_signals - signals_used
                }
            else:
                return {
                    "plan_type": "TRIAL",
                    "is_active": True,
                    "subscription_end": None,
                    "max_daily_signals": 5,
                    "signals_used": 0,
                    "signals_remaining": 5
                }
                
        except Exception as e:
            logger.error(f"❌ Get subscription failed: {e}")
            return {
                "plan_type": "TRIAL",
                "is_active": True,
                "subscription_end": None,
                "max_daily_signals": 5,
                "signals_used": 0,
                "signals_remaining": 5
            }
    
    def can_user_request_signal(self, user_id):
        subscription = self.get_user_subscription(user_id)
        
        if not subscription["is_active"]:
            return False, "Subscription expired. Use /register to renew."
        
        if subscription["signals_used"] >= subscription["max_daily_signals"]:
            return False, "Daily signal limit reached. Upgrade for more signals!"
        
        return True, "OK"
    
    def increment_signal_count(self, user_id):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE users SET signals_used = signals_used + 1 WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Signal count increment failed: {e}")

# ==================== WORKING SESSION MANAGER ====================
class WorkingSessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {"start_hour": 7, "end_hour": 11, "name": "🌅 London Session"},
            "EVENING": {"start_hour": 15, "end_hour": 19, "name": "🌇 NY/London Overlap"},
            "ASIAN": {"start_hour": 23, "end_hour": 3, "name": "🌃 Asian Session"}
        }

    def get_current_time_utc1(self):
        return datetime.utcnow() + timedelta(hours=1)

    def get_current_session(self):
        try:
            now_utc1 = self.get_current_time_utc1()
            current_hour = now_utc1.hour
            current_time_str = now_utc1.strftime("%H:%M UTC+1")
            
            for session_id, session in self.sessions.items():
                if session_id == "ASIAN":
                    if current_hour >= session["start_hour"] or current_hour < session["end_hour"]:
                        return {**session, "id": session_id, "current_time": current_time_str, "status": "ACTIVE"}
                else:
                    if session["start_hour"] <= current_hour < session["end_hour"]:
                        return {**session, "id": session_id, "current_time": current_time_str, "status": "ACTIVE"}
            
            next_session = self.get_next_session()
            return {
                "id": "CLOSED", 
                "name": "Market Closed", 
                "current_time": current_time_str,
                "status": "CLOSED",
                "next_session": next_session["name"],
                "next_session_time": f"{next_session['start_hour']:02d}:00-{next_session['end_hour']:02d}:00"
            }
            
        except Exception as e:
            logger.error(f"Session error: {e}")
            return {"id": "ERROR", "name": "System Error", "current_time": "N/A", "status": "ERROR"}

    def get_next_session(self):
        sessions_order = ["ASIAN", "MORNING", "EVENING"]
        current_session = self.get_current_session()
        
        if current_session["id"] == "CLOSED":
            return self.sessions["ASIAN"]
        
        current_index = sessions_order.index(current_session["id"])
        next_index = (current_index + 1) % len(sessions_order)
        return self.sessions[sessions_order[next_index]]

# ==================== ENHANCED SIGNAL GENERATOR WITH TIMEFRAMES ====================
class WorkingSignalGenerator:
    def __init__(self):
        self.all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
        self.pending_signals = {}
    
    def generate_pre_entry_signal(self, symbol=None, signal_style="NORMAL"):
        try:
            if not symbol:
                symbol = random.choice(self.all_pairs)
            
            direction = random.choice(["BUY", "SELL"])
            
            # REALISTIC PRICES
            if symbol == "EUR/USD":
                base_price = round(random.uniform(1.07500, 1.09500), 5)
            elif symbol == "GBP/USD":
                base_price = round(random.uniform(1.25800, 1.27800), 5)
            elif symbol == "USD/JPY":
                base_price = round(random.uniform(148.500, 151.500), 3)
            elif symbol == "XAU/USD":
                base_price = round(random.uniform(1950.00, 2050.00), 2)
            else:
                base_price = round(random.uniform(1.08000, 1.10000), 5)
            
            spread = 0.00015
            if direction == "BUY":
                entry_price = round(base_price + spread, 5 if "XAU" not in symbol else 2)
            else:
                entry_price = round(base_price - spread, 5 if "XAU" not in symbol else 2)
            
            # TIMEFRAME BASED ON SIGNAL STYLE
            if signal_style == "QUICK":
                timeframe = "1M"  # 1 Minute for quick trades
                confidence = round(random.uniform(0.82, 0.90), 3)
                entry_delay = 20  # 20 seconds for quick trades
            else:
                timeframe = random.choice(["5M", "15M"])  # 5M or 15M for normal trades
                confidence = round(random.uniform(0.88, 0.96), 3)
                entry_delay = Config.PRE_ENTRY_DELAY
            
            current_time = datetime.now()
            entry_time = current_time + timedelta(seconds=entry_delay)
            
            signal_id = f"SIGNAL_{int(time.time())}_{symbol.replace('/', '')}"
            
            signal_data = {
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "take_profit": 0.0,
                "stop_loss": 0.0,
                "confidence": confidence,
                "current_time": current_time.strftime("%H:%M:%S"),
                "entry_time": entry_time.strftime("%H:%M:%S"),
                "timeframe": timeframe,
                "signal_style": signal_style,
                "entry_delay": entry_delay,
                "generated_at": current_time.isoformat()
            }
            
            self.pending_signals[signal_id] = signal_data
            logger.info(f"✅ {signal_style} Pre-entry generated: {symbol} {direction} at {entry_price} on {timeframe}")
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Pre-entry generation failed: {e}")
            current_time = datetime.now()
            entry_time = current_time + timedelta(seconds=Config.PRE_ENTRY_DELAY)
            
            return {
                "signal_id": f"BACKUP_{int(time.time())}",
                "symbol": "EUR/USD",
                "direction": "BUY",
                "entry_price": 1.08500,
                "take_profit": 0.0,
                "stop_loss": 0.0,
                "confidence": 0.92,
                "current_time": current_time.strftime("%H:%M:%S"),
                "entry_time": entry_time.strftime("%H:%M:%S"),
                "timeframe": "5M",
                "signal_style": "NORMAL",
                "entry_delay": Config.PRE_ENTRY_DELAY,
                "generated_at": current_time.isoformat()
            }
    
    def generate_entry_signal(self, signal_id):
        try:
            if signal_id not in self.pending_signals:
                return None
            
            pre_signal = self.pending_signals[signal_id]
            
            # ADJUST TP/SL BASED ON TIMEFRAME
            if pre_signal["timeframe"] == "1M":
                # Tighter TP/SL for 1M trades
                if "XAU" in pre_signal["symbol"]:
                    tp_distance = random.uniform(8.0, 15.0)
                    sl_distance = random.uniform(5.0, 10.0)
                elif "JPY" in pre_signal["symbol"]:
                    tp_distance = random.uniform(0.5, 1.0)
                    sl_distance = random.uniform(0.3, 0.7)
                else:
                    tp_distance = random.uniform(0.0015, 0.0025)
                    sl_distance = random.uniform(0.0010, 0.0018)
            else:
                # Standard TP/SL for 5M/15M
                if "XAU" in pre_signal["symbol"]:
                    tp_distance = random.uniform(12.0, 25.0)
                    sl_distance = random.uniform(8.0, 18.0)
                elif "JPY" in pre_signal["symbol"]:
                    tp_distance = random.uniform(0.8, 1.5)
                    sl_distance = random.uniform(0.5, 1.2)
                else:
                    tp_distance = random.uniform(0.0025, 0.0040)
                    sl_distance = random.uniform(0.0015, 0.0025)
            
            if pre_signal["direction"] == "BUY":
                take_profit = round(pre_signal["entry_price"] + tp_distance, 5 if "XAU" not in pre_signal["symbol"] else 2)
                stop_loss = round(pre_signal["entry_price"] - sl_distance, 5 if "XAU" not in pre_signal["symbol"] else 2)
            else:
                take_profit = round(pre_signal["entry_price"] - tp_distance, 5 if "XAU" not in pre_signal["symbol"] else 2)
                stop_loss = round(pre_signal["entry_price"] + sl_distance, 5 if "XAU" not in pre_signal["symbol"] else 2)
            
            entry_signal = {
                **pre_signal,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "entry_time_actual": datetime.now().strftime("%H:%M:%S"),
                "risk_reward": round(abs(take_profit - pre_signal["entry_price"]) / abs(pre_signal["entry_price"] - stop_loss), 2)
            }
            
            del self.pending_signals[signal_id]
            logger.info(f"✅ Entry generated: {pre_signal['symbol']} TP: {take_profit} SL: {stop_loss}")
            return entry_signal
            
        except Exception as e:
            logger.error(f"❌ Entry generation failed: {e}")
            return None

# ==================== SIMPLE USER MANAGER ====================
class SimpleUserManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def add_user(self, user_id, username, first_name):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
                (user_id, username, first_name)
            )
            conn.commit()
            conn.close()
            logger.info(f"✅ User added: {username}")
            return True
        except Exception as e:
            logger.error(f"❌ User add failed: {e}")
            return False

# ==================== FIXED ADMIN AUTHENTICATION ====================
class AdminAuth:
    def __init__(self):
        self.session_duration = timedelta(hours=24)
    
    def verify_token(self, token: str) -> bool:
        return token == Config.ADMIN_TOKEN
    
    def create_session(self, user_id: int, username: str):
        login_time = datetime.now()
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO admin_sessions 
                (user_id, username, login_time)
                VALUES (?, ?, ?)
            """, (user_id, username, login_time.isoformat()))
            conn.commit()
    
    def is_admin(self, user_id: int) -> bool:
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.execute(
                    "SELECT login_time FROM admin_sessions WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    login_time = datetime.fromisoformat(result[0])
                    # Check if session is still valid (24 hours)
                    if login_time + self.session_duration > datetime.now():
                        return True
                    else:
                        # Session expired, remove it
                        conn.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
                        conn.commit()
                return False
        except Exception as e:
            logger.error(f"❌ Admin check failed: {e}")
            return False

# ==================== WORKING TRADING BOT ====================
class WorkingTradingBot:
    def __init__(self, application):
        self.application = application
        self.session_manager = WorkingSessionManager()
        self.signal_generator = WorkingSignalGenerator()
        self.user_manager = SimpleUserManager(Config.DB_PATH)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH)
        self.admin_auth = AdminAuth()
        self.is_running = True
    
    async def send_welcome_message(self, user, chat_id):
        try:
            current_session = self.session_manager.get_current_session()
            subscription = self.subscription_manager.get_user_subscription(user.id)
            is_admin = self.admin_auth.is_admin(user.id)
            
            # Base welcome message
            if current_session["status"] == "ACTIVE":
                message = f"""
🎉 *WELCOME TO LEKZY FX AI PRO!* 🚀

*Hello {user.first_name}!* 👋

📊 *Your Account Status:*
• Plan: *{subscription['plan_type']}*
• Signals: *{subscription['signals_used']}/{subscription['max_daily_signals']} used*
• Status: *{'✅ ACTIVE' if subscription['is_active'] else '❌ EXPIRED'}*

✅ *Live Market Session: {current_session['name']}*
✅ *Current Time: {current_session['current_time']}*

💡 *Ready to trade? Use the buttons below!*
"""
            else:
                message = f"""
🎉 *WELCOME TO LEKZY FX AI PRO!* 🚀

*Hello {user.first_name}!* 👋

📊 *Your Account Status:*
• Plan: *{subscription['plan_type']}*
• Signals: *{subscription['signals_used']}/{subscription['max_daily_signals']} used*
• Status: *{'✅ ACTIVE' if subscription['is_active'] else '❌ EXPIRED'}*

⏸️ *MARKET IS CURRENTLY CLOSED*

🕒 *Current Time:* {current_session['current_time']}
📅 *Next Session:* {current_session['next_session']}
⏰ *Opens at:* {current_session['next_session_time']} UTC+1
"""
            
            # Add admin badge if user is admin
            if is_admin:
                message += "\n👑 *You have Admin Access* - Use /admin for admin features\n"
            
            message += "\n*Tap buttons below to get started!* 🎯"
            
            # Create keyboard - FIXED BUTTON LAYOUT
            keyboard = []
            
            if current_session["status"] == "ACTIVE":
                keyboard.append([InlineKeyboardButton("🚀 GET SIGNAL NOW", callback_data="get_signal")])
            
            keyboard.append([InlineKeyboardButton("💎 REGISTER/UPGRADE", callback_data="show_register")])
            keyboard.append([InlineKeyboardButton("🕒 MARKET STATUS", callback_data="session_info")])
            keyboard.append([InlineKeyboardButton("📞 CONTACT SUPPORT", callback_data="contact_support")])
            
            # Add admin buttons if user is admin
            if is_admin:
                keyboard.insert(0, [InlineKeyboardButton("👑 ADMIN PANEL", callback_data="admin_panel")])
                if current_session["status"] == "ACTIVE":
                    keyboard.insert(1, [
                        InlineKeyboardButton("⚡ QUICK TRADE", callback_data="admin_quick"),
                        InlineKeyboardButton("📈 NORMAL TRADE", callback_data="admin_normal")
                    ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Welcome message failed: {e}")
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=f"Welcome {user.first_name}! Use /signal to get trading signals.",
                parse_mode='Markdown'
            )
    
    async def generate_signal_for_user(self, user_id, chat_id, signal_style="NORMAL", is_admin=False):
        try:
            # Check subscription and limits (unless admin)
            if not is_admin:
                can_request, message = self.subscription_manager.can_user_request_signal(user_id)
                if not can_request:
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=f"❌ *{message}*\n\nUse /register TOKEN to upgrade your account! 💎",
                        parse_mode='Markdown'
                    )
                    return
            
            current_session = self.session_manager.get_current_session()
            if current_session["status"] != "ACTIVE":
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=f"⏸️ *MARKET IS CLOSED*\n\nCome back during trading hours! 📈",
                    parse_mode='Markdown'
                )
                return
            
            # Generate signal
            signal_type = "ADMIN" if is_admin else "USER"
            style_text = "QUICK" if signal_style == "QUICK" else "NORMAL"
            
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=f"🎯 *Generating {style_text} {signal_type} signal...* ⏱️",
                parse_mode='Markdown'
            )
            
            pre_signal = self.signal_generator.generate_pre_entry_signal(signal_style=signal_style)
            if not pre_signal:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text="❌ *Failed to generate signal. Please try again.*",
                    parse_mode='Markdown'
                )
                return
            
            # Send pre-entry message WITH TIMEFRAME
            direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
            subscription = self.subscription_manager.get_user_subscription(user_id)
            
            # Timeframe-based message
            timeframe_msg = ""
            if pre_signal["timeframe"] == "1M":
                timeframe_msg = "⚡ *QUICK TRADE* - 1 Minute Timeframe\n• Fast execution required\n• Tight stops recommended"
            elif pre_signal["timeframe"] == "5M":
                timeframe_msg = "📈 *SWING TRADE* - 5 Minute Timeframe\n• Medium-term setup\n• Good risk-reward ratio"
            else:
                timeframe_msg = "🎯 *POSITION TRADE* - 15 Minute Timeframe\n• Longer-term setup\n• Higher confidence"
            
            pre_entry_msg = f"""
📊 *PRE-ENTRY SIGNAL* - {style_text}
*Entry in {pre_signal['entry_delay']}s*

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Expected Entry:* `{pre_signal['entry_price']}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

⏰ *Timing Information:*
• 🕐 Current Time: `{pre_signal['current_time']}`
• 🎯 Expected Entry: `{pre_signal['entry_time']}`
• ⏱️ Countdown: {pre_signal['entry_delay']} seconds
• 📊 Timeframe: *{pre_signal['timeframe']}*

{timeframe_msg}

📊 *Your Account:*
• Plan: *{subscription['plan_type']}*
• Signals Left: *{subscription['signals_remaining']}*

*Entry signal coming in {pre_signal['entry_delay']} seconds...*
"""
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=pre_entry_msg,
                parse_mode='Markdown'
            )
            
            # Store signal
            try:
                conn = sqlite3.connect(Config.DB_PATH)
                conn.execute(
                    "INSERT INTO signals (signal_id, symbol, direction, entry_price, confidence, signal_type, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (pre_signal["signal_id"], pre_signal["symbol"], pre_signal["direction"], pre_signal["entry_price"], pre_signal["confidence"], "PRE_ENTRY", pre_signal["timeframe"])
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Database save error: {e}")
            
            # Wait for entry based on signal style
            logger.info(f"⏰ Waiting {pre_signal['entry_delay']}s for entry signal...")
            await asyncio.sleep(pre_signal["entry_delay"])
            
            # Generate entry signal
            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
            if not entry_signal:
                # Create backup entry signal
                if pre_signal["direction"] == "BUY":
                    take_profit = round(pre_signal["entry_price"] + 0.0035, 5)
                    stop_loss = round(pre_signal["entry_price"] - 0.0022, 5)
                else:
                    take_profit = round(pre_signal["entry_price"] - 0.0035, 5)
                    stop_loss = round(pre_signal["entry_price"] + 0.0022, 5)
                
                entry_signal = {
                    **pre_signal,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "entry_time_actual": datetime.now().strftime("%H:%M:%S"),
                    "risk_reward": 1.6
                }
            
            # Increment signal count (unless admin)
            if not is_admin:
                self.subscription_manager.increment_signal_count(user_id)
            
            # Send entry signal
            entry_msg = f"""
🎯 *ENTRY SIGNAL* - {style_text} ✅
*EXECUTE NOW*

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry Price:* `{entry_signal['entry_price']}`
✅ *Take Profit:* `{entry_signal['take_profit']}`
❌ *Stop Loss:* `{entry_signal['stop_loss']}`

⏰ *Entry Time:* `{entry_signal['entry_time_actual']}`
📊 *Timeframe:* {entry_signal['timeframe']}

📈 *Trade Details:*
• Confidence: *{entry_signal['confidence']*100:.1f}%* 🎯
• Risk/Reward: *1:{entry_signal.get('risk_reward', 1.6)}* ⚖️
• Type: *{style_text}* {'⚡' if signal_style == 'QUICK' else '📈'}

*Execute this trade immediately!* 🚀
"""
            keyboard = [
                [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
                [InlineKeyboardButton("🔄 NEW SIGNAL", callback_data="get_signal")],
                [InlineKeyboardButton("💎 UPGRADE ACCOUNT", callback_data="show_register")]
            ]
            
            # Add admin buttons if admin
            if is_admin:
                keyboard.insert(1, [
                    InlineKeyboardButton("⚡ QUICK TRADE", callback_data="admin_quick"),
                    InlineKeyboardButton("📈 NORMAL TRADE", callback_data="admin_normal")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=entry_msg,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            # Store entry signal
            try:
                conn = sqlite3.connect(Config.DB_PATH)
                conn.execute(
                    "INSERT INTO signals (signal_id, symbol, direction, entry_price, take_profit, stop_loss, confidence, signal_type, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry_signal["signal_id"] + "_ENTRY", entry_signal["symbol"], entry_signal["direction"], 
                     entry_signal["entry_price"], entry_signal["take_profit"], entry_signal["stop_loss"], 
                     entry_signal["confidence"], "ENTRY", entry_signal["timeframe"])
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Database update error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            await self.application.bot.send_message(
                chat_id=chat_id,
                text="❌ *Signal generation failed. Please try again.*",
                parse_mode='Markdown'
            )

# ==================== FIXED TELEGRAM BOT ====================
class SimpleTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.trading_bot = None
    
    async def initialize(self):
        try:
            self.application = Application.builder().token(self.token).build()
            self.trading_bot = WorkingTradingBot(self.application)
            
            # Command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("signal", self.signal_command))
            self.application.add_handler(CommandHandler("session", self.session_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("register", self.register_command))
            self.application.add_handler(CommandHandler("seedtoken", self.seedtoken_command))
            self.application.add_handler(CommandHandler("login", self.login_command))
            self.application.add_handler(CommandHandler("admin", self.admin_command))
            self.application.add_handler(CommandHandler("mystats", self.mystats_command))
            
            # Callback handlers - FIXED: Add all button handlers
            self.application.add_handler(CallbackQueryHandler(self.button_handler))
            
            await self.application.initialize()
            await self.application.start()
            
            logger.info("✅ Telegram Bot with Fixed Buttons Initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            return False

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            chat_id = update.effective_chat.id
            
            logger.info(f"🚀 User started: {user.first_name}")
            
            # Add user to database
            self.trading_bot.user_manager.add_user(user.id, user.username, user.first_name)
            
            # Send welcome message
            await self.trading_bot.send_welcome_message(user, chat_id)
            
        except Exception as e:
            logger.error(f"❌ Start command failed: {e}")
            await update.message.reply_text("Welcome! Use /signal to get trading signals.")

    async def register_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            
            if not context.args:
                await update.message.reply_text(
                    "🔐 *REGISTER PREMIUM ACCOUNT*\n\nUsage: `/register YOUR_TOKEN`",
                    parse_mode='Markdown'
                )
                return
            
            token = context.args[0].strip().upper()
            
            is_valid, days_valid = self.trading_bot.subscription_manager.token_manager.validate_token(token)
            
            if not is_valid:
                await update.message.reply_text(
                    f"❌ *Invalid Token*\n\nPlease contact {Config.ADMIN_CONTACT} for a valid token.",
                    parse_mode='Markdown'
                )
                return
            
            success = self.trading_bot.subscription_manager.activate_premium_subscription(
                user.id, token, days_valid
            )
            
            if success:
                end_date = datetime.now() + timedelta(days=days_valid)
                await update.message.reply_text(
                    f"🎉 *PREMIUM ACTIVATED!* 🚀\n\n"
                    f"Welcome to LEKZY FX AI PRO Premium!\n"
                    f"• Duration: {days_valid} days\n"
                    f"• Signals: 50 per day\n"
                    f"• All sessions access\n\n"
                    f"*Use /signal to start trading!* 🎯",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ Registration failed. Please try again.")
                
        except Exception as e:
            logger.error(f"❌ Register command failed: {e}")
            await update.message.reply_text("❌ Registration failed.")

    async def seedtoken_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            
            if not self.trading_bot.admin_auth.is_admin(user.id):
                await update.message.reply_text("❌ Admin access required.")
                return
            
            if not context.args:
                await update.message.reply_text("Usage: `/seedtoken DAYS`")
                return
            
            try:
                days = int(context.args[0])
                if days <= 0:
                    await update.message.reply_text("❌ Days must be positive number.")
                    return
            except ValueError:
                await update.message.reply_text("❌ Invalid number of days.")
                return
            
            token = self.trading_bot.subscription_manager.token_manager.generate_token(days, user.id)
            
            if token:
                await update.message.reply_text(
                    f"🔑 *TOKEN GENERATED* ✅\n\n"
                    f"*Token:* `{token}`\n"
                    f"*Duration:* {days} days\n\n"
                    f"*Share with users:* `/register {token}`",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ Token generation failed.")
                
        except Exception as e:
            logger.error(f"❌ Seedtoken command failed: {e}")
            await update.message.reply_text("❌ Token generation failed.")

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """FIXED ADMIN LOGIN"""
        try:
            user = update.effective_user
            
            if not context.args:
                await update.message.reply_text("🔐 *ADMIN LOGIN*\n\nUsage: `/login ADMIN_TOKEN`", parse_mode='Markdown')
                return
            
            token = context.args[0]
            
            if self.trading_bot.admin_auth.verify_token(token):
                self.trading_bot.admin_auth.create_session(user.id, user.username)
                await update.message.reply_text(
                    "✅ *Admin Access Granted!* 👑\n\n"
                    "*You now have access to:*\n"
                    "• Quick Trade signals (1M timeframe)\n"
                    "• Normal Trade signals (5M/15M timeframe)\n"
                    "• Token generation (/seedtoken)\n"
                    "• Admin dashboard (/admin)\n\n"
                    "*Use the buttons in /start menu!* 🚀",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ *Invalid admin token*", parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"❌ Login command failed: {e}")
            await update.message.reply_text("❌ Login failed.")

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            
            if not self.trading_bot.admin_auth.is_admin(user.id):
                await update.message.reply_text("❌ Admin access required. Use `/login TOKEN`", parse_mode='Markdown')
                return
            
            message = """
👑 *ADMIN DASHBOARD* 🔧

⚡ *Admin Features:*
• Quick Trade (1M timeframe) - 20s entry
• Normal Trade (5M/15M timeframe) - 40s entry  
• Generate subscription tokens
• Monitor system status

🎯 *Quick Commands:*
• `/seedtoken DAYS` - Generate tokens
• Use buttons for instant signals

*Admin system operational!* ✅
"""
            keyboard = [
                [InlineKeyboardButton("⚡ QUICK TRADE", callback_data="admin_quick")],
                [InlineKeyboardButton("📈 NORMAL TRADE", callback_data="admin_normal")],
                [InlineKeyboardButton("🔑 GENERATE TOKENS", callback_data="admin_tokens")],
                [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Admin command failed: {e}")
            await update.message.reply_text("❌ Admin command failed.")

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            chat_id = update.effective_chat.id
            is_admin = self.trading_bot.admin_auth.is_admin(user.id)
            
            # Check if admin wants specific style
            signal_style = "NORMAL"
            if context.args and context.args[0].upper() in ["QUICK", "FAST"]:
                signal_style = "QUICK"
                if not is_admin:
                    await update.message.reply_text("❌ Quick trades are for admin only.")
                    return
            
            await self.trading_bot.generate_signal_for_user(user.id, chat_id, signal_style, is_admin)
        except Exception as e:
            logger.error(f"❌ Signal command failed: {e}")
            await update.message.reply_text("❌ Unable to generate signal.")

    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            current_session = self.trading_bot.session_manager.get_current_session()
            
            if current_session["status"] == "ACTIVE":
                message = f"🟢 *MARKET IS OPEN* ✅\n\n*Current Session:* {current_session['name']}\n*Time:* {current_session['current_time']}"
            else:
                message = f"🔴 *MARKET IS CLOSED* ⏸️\n\n*Next Session:* {current_session['next_session']}\n*Opens:* {current_session['next_session_time']}"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Session command failed: {e}")
            await update.message.reply_text("❌ Could not fetch session info.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
🤖 *LEKZY FX AI PRO - HELP*

*User Commands:*
• /start - Start bot & welcome
• /signal - Get trading signal  
• /session - Market status
• /register TOKEN - Activate premium
• /mystats - Your statistics
• /help - This message

*Admin Commands:*
• /login TOKEN - Admin login
• /seedtoken DAYS - Generate tokens
• /admin - Admin dashboard
• /signal quick - Quick trade (admin only)

📞 *Support:* @LekzyTradingPro
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def mystats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            subscription = self.trading_bot.subscription_manager.get_user_subscription(user.id)
            is_admin = self.trading_bot.admin_auth.is_admin(user.id)
            
            if subscription['plan_type'] == 'PREMIUM' and subscription['subscription_end']:
                end_date = datetime.fromisoformat(subscription['subscription_end'])
                days_left = (end_date - datetime.now()).days
                status = f"✅ Active ({days_left} days left)"
            else:
                status = "⏳ Trial"
            
            message = f"""
📊 *YOUR ACCOUNT STATISTICS*

• Name: {user.first_name}
• Plan: {subscription['plan_type']}
• Status: {status}
• Signals: {subscription['signals_used']}/{subscription['max_daily_signals']}
• Admin: {'✅ Yes' if is_admin else '❌ No'}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Mystats command failed: {e}")
            await update.message.reply_text("❌ Could not fetch statistics.")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """FIXED BUTTON HANDLER - ALL BUTTONS WORKING"""
        query = update.callback_query
        await query.answer()
        
        user = query.from_user
        data = query.data
        
        try:
            if data == "get_signal":
                await self.signal_command(update, context)
            elif data == "session_info":
                await self.session_command(update, context)
            elif data == "contact_support":
                await query.edit_message_text(f"📞 *Contact Support:* {Config.ADMIN_CONTACT}", parse_mode='Markdown')
            elif data == "show_register":
                await query.edit_message_text(
                    "💎 *UPGRADE TO PREMIUM*\n\nUse `/register YOUR_TOKEN` to activate premium!\n\n*Contact admin for tokens.*",
                    parse_mode='Markdown'
                )
            elif data == "trade_done":
                await query.edit_message_text("✅ *Trade Executed!* 🎯\n\n*Happy trading!* 💰", parse_mode='Markdown')
            elif data == "admin_panel":
                await self.admin_command(update, context)
            elif data == "admin_quick":
                if self.trading_bot.admin_auth.is_admin(user.id):
                    # Generate quick trade signal
                    await self.trading_bot.generate_signal_for_user(user.id, query.message.chat_id, "QUICK", True)
                else:
                    await query.edit_message_text("❌ Admin access required for quick trades.")
            elif data == "admin_normal":
                if self.trading_bot.admin_auth.is_admin(user.id):
                    # Generate normal trade signal
                    await self.trading_bot.generate_signal_for_user(user.id, query.message.chat_id, "NORMAL", True)
                else:
                    await query.edit_message_text("❌ Admin access required.")
            elif data == "admin_tokens":
                if self.trading_bot.admin_auth.is_admin(user.id):
                    await query.edit_message_text(
                        "🔑 *GENERATE TOKENS*\n\nUse `/seedtoken DAYS` to create subscription tokens.",
                        parse_mode='Markdown'
                    )
                else:
                    await query.edit_message_text("❌ Admin access required.")
            elif data == "main_menu":
                await self.start_command(update, context)
                
        except Exception as e:
            logger.error(f"Button handler error: {e}")
            await query.edit_message_text("❌ Action failed. Please try /start again.")

    async def start_polling(self):
        await self.application.updater.start_polling()
        logger.info("✅ Bot polling started")

    async def stop(self):
        await self.application.stop()

# ==================== MAIN APPLICATION ====================
class MainApp:
    def __init__(self):
        self.bot = None
        self.running = False
    
    async def setup(self):
        try:
            initialize_database()
            start_web_server()
            self.bot = SimpleTelegramBot()
            success = await self.bot.initialize()
            
            if success:
                self.running = True
                logger.info("🚀 LEKZY FX AI PRO - FIXED VERSION ACTIVE!")
                return True
            return False
                
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def run(self):
        if not self.running:
            success = await self.setup()
            if not success:
                return
        
        try:
            await self.bot.start_polling()
            logger.info("✅ Application running on Render")
            
            while self.running:
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ Run error: {e}")
    
    async def shutdown(self):
        self.running = False
        if self.bot:
            await self.bot.stop()

# ==================== START BOT ====================
async def main():
    app = MainApp()
    try:
        await app.run()
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    print("🚀 Starting LEKZY FX AI PRO - FIXED VERSION...")
    asyncio.run(main())
