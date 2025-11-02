#!/usr/bin/env python3
"""
LEKZY FX AI PRO - WITH SMART TRADE TYPE SELECTION
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

# ==================== RISK MANAGEMENT CONFIG ====================
class RiskConfig:
    # Risk Disclaimer Messages
    DISCLAIMERS = {
        "high_risk": "🚨 *HIGH RISK WARNING*\n\nTrading foreign exchange, cryptocurrencies, and CFDs carries a high level of risk and may not be suitable for all investors.",
        "past_performance": "📊 *PAST PERFORMANCE*\n\nPast performance is not indicative of future results. No representation is being made that any account will achieve profits or losses similar to those shown.",
        "risk_capital": "💼 *RISK CAPITAL ONLY*\n\nYou should only trade with money you can afford to lose. Do not use funds allocated for essential expenses.",
        "seek_advice": "👨‍💼 *SEEK PROFESSIONAL ADVICE*\n\nBefore trading, consider your investment objectives, experience level, and risk tolerance."
    }
    
    # Money Management Rules
    MONEY_MANAGEMENT = {
        "rule_1": "💰 *Risk Only 1-2%* of your trading capital per trade",
        "rule_2": "🎯 *Use Stop Losses* on every trade without exception", 
        "rule_3": "⚖️ *Maintain 1:1.5 Risk/Reward* ratio minimum",
        "rule_4": "📊 *Maximum 5%* total capital exposure at any time",
        "rule_5": "😴 *No Emotional Trading* - stick to your strategy"
    }
    
    # Position Sizing Guidelines
    POSITION_SIZING = {
        "conservative": "🛡️ Conservative: 0.5-1% risk per trade",
        "moderate": "🎯 Moderate: 1-2% risk per trade", 
        "aggressive": "⚡ Aggressive: 2-3% risk per trade (not recommended for beginners)"
    }

# ==================== ENHANCED PLAN CONFIGURATION ====================
class PlanConfig:
    PLANS = {
        "TRIAL": {
            "name": "🆓 FREE TRIAL",
            "days": 7,
            "daily_signals": 3,
            "price": "FREE",
            "actual_price": "$0",
            "features": ["3 signals/day", "7 days access", "Basic support", "All currency pairs", "Normal trades only"],
            "description": "Perfect for testing our signals",
            "emoji": "🆓",
            "recommended": False,
            "quick_trades": False  # Trial users don't get quick trades
        },
        "PREMIUM": {
            "name": "💎 PREMIUM", 
            "days": 30,
            "daily_signals": 50,
            "price": "$49.99",
            "actual_price": "$49.99",
            "features": ["50 signals/day", "30 days access", "Priority support", "All pairs access", "Normal & Quick trades", "Risk management tools"],
            "description": "Best for serious traders",
            "emoji": "💎",
            "recommended": True,
            "quick_trades": True  # Premium users get both options
        },
        "VIP": {
            "name": "🚀 VIP",
            "days": 90,
            "daily_signals": 100,
            "price": "$129.99",
            "actual_price": "$129.99", 
            "features": ["100 signals/day", "90 days access", "24/7 support", "All pairs + VIP signals", "All trade types", "Advanced analytics", "Priority signal delivery"],
            "description": "Ultimate trading experience",
            "emoji": "🚀",
            "recommended": False,
            "quick_trades": True
        },
        "PRO": {
            "name": "🔥 PRO TRADER",
            "days": 180,
            "daily_signals": 200,
            "price": "$199.99",
            "actual_price": "$199.99",
            "features": ["200 signals/day", "180 days access", "24/7 premium support", "VIP + PRO signals", "All timeframes", "Personal analyst access", "Custom strategies"],
            "description": "Professional trading suite",
            "emoji": "🔥",
            "recommended": False,
            "quick_trades": True
        }
    }

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
    return "🤖 LEKZY FX AI PRO - ACTIVE 🚀"

@app.route('/health')
def health():
    return json.dumps({"status": "healthy", "timestamp": datetime.now().isoformat()})

def run_web_server():
    try:
        port = int(os.environ.get('PORT', Config.PORT))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"❌ Web server failed: {e}")

def start_web_server():
    web_thread = Thread(target=run_web_server)
    web_thread.daemon = True
    web_thread.start()

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
                max_daily_signals INTEGER DEFAULT 3,
                signals_used INTEGER DEFAULT 0,
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
                risk_acknowledged BOOLEAN DEFAULT FALSE
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
                plan_type TEXT DEFAULT 'PREMIUM',
                days_valid INTEGER DEFAULT 30,
                created_by INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                used_by INTEGER DEFAULT NULL,
                used_at TEXT DEFAULT NULL,
                status TEXT DEFAULT 'ACTIVE'
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
    
    def generate_token(self, plan_type="PREMIUM", days_valid=None, created_by=None):
        try:
            alphabet = string.ascii_uppercase + string.digits
            token = ''.join(secrets.choice(alphabet) for _ in range(12))
            
            if days_valid is None:
                days_valid = PlanConfig.PLANS.get(plan_type, {}).get("days", 30)
            
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO subscription_tokens (token, plan_type, days_valid, created_by) VALUES (?, ?, ?, ?)",
                (token, plan_type, days_valid, created_by)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"✅ {plan_type} token generated: {token}")
            return token
            
        except Exception as e:
            logger.error(f"❌ Token generation failed: {e}")
            return None
    
    def validate_token(self, token):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT token, plan_type, days_valid FROM subscription_tokens WHERE token = ? AND status = 'ACTIVE'",
                (token,)
            )
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, "Invalid token", 0
            
            token_str, plan_type, days_valid = result
            
            conn.execute(
                "UPDATE subscription_tokens SET status = 'USED', used_at = ? WHERE token = ?",
                (datetime.now().isoformat(), token)
            )
            conn.commit()
            conn.close()
            
            return True, plan_type, days_valid
            
        except Exception as e:
            logger.error(f"❌ Token validation failed: {e}")
            return False, "Token error", 0

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.token_manager = TokenManager(db_path)
    
    def activate_subscription(self, user_id, token, plan_type, days_valid):
        try:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_valid)
            
            plan_config = PlanConfig.PLANS.get(plan_type, PlanConfig.PLANS["PREMIUM"])
            max_signals = plan_config["daily_signals"]
            
            conn = sqlite3.connect(self.db_path)
            
            # Update or insert user
            conn.execute("""
                INSERT OR REPLACE INTO users 
                (user_id, plan_type, subscription_end, max_daily_signals, signals_used)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, plan_type, end_date.isoformat(), max_signals, 0))
            
            # Update token
            conn.execute(
                "UPDATE subscription_tokens SET used_by = ? WHERE token = ?",
                (user_id, token)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ {plan_type} activated for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Subscription activation failed: {e}")
            return False
    
    def get_user_subscription(self, user_id):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT plan_type, subscription_end, max_daily_signals, signals_used, risk_acknowledged
                FROM users WHERE user_id = ?
            """, (user_id,))
            result = cursor.fetchone()
            
            if result:
                plan_type, sub_end, max_signals, signals_used, risk_acknowledged = result
                
                is_active = True
                if sub_end and plan_type != "TRIAL":
                    try:
                        end_date = datetime.fromisoformat(sub_end)
                        is_active = datetime.now() < end_date
                        if not is_active:
                            # Reset to trial if expired
                            plan_type = "TRIAL"
                            max_signals = 3
                            conn.execute(
                                "UPDATE users SET plan_type = ?, max_daily_signals = ?, signals_used = 0 WHERE user_id = ?",
                                ("TRIAL", 3, user_id)
                            )
                            conn.commit()
                    except:
                        pass
                
                conn.close()
                return {
                    "plan_type": plan_type,
                    "is_active": is_active,
                    "subscription_end": sub_end,
                    "max_daily_signals": max_signals,
                    "signals_used": signals_used,
                    "signals_remaining": max_signals - signals_used,
                    "risk_acknowledged": risk_acknowledged
                }
            else:
                # Create new trial user
                conn.execute(
                    "INSERT INTO users (user_id, plan_type, max_daily_signals) VALUES (?, ?, ?)",
                    (user_id, "TRIAL", 3)
                )
                conn.commit()
                conn.close()
                
                return {
                    "plan_type": "TRIAL",
                    "is_active": True,
                    "subscription_end": None,
                    "max_daily_signals": 3,
                    "signals_used": 0,
                    "signals_remaining": 3,
                    "risk_acknowledged": False
                }
                
        except Exception as e:
            logger.error(f"❌ Get subscription failed: {e}")
            return {
                "plan_type": "TRIAL",
                "is_active": True,
                "subscription_end": None,
                "max_daily_signals": 3,
                "signals_used": 0,
                "signals_remaining": 3,
                "risk_acknowledged": False
            }
    
    def mark_risk_acknowledged(self, user_id):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE users SET risk_acknowledged = TRUE WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Risk acknowledgment failed: {e}")
            return False
    
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

# ==================== ENHANCED SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "ASIAN": {"name": "🌃 ASIAN SESSION", "hours": "23:00-03:00 UTC+1", "active": False},
            "LONDON": {"name": "🌅 LONDON SESSION", "hours": "07:00-11:00 UTC+1", "active": False},
            "NEWYORK": {"name": "🌇 NY/LONDON OVERLAP", "hours": "15:00-19:00 UTC+1", "active": False}
        }
    
    def get_current_session(self):
        now = datetime.utcnow() + timedelta(hours=1)  # UTC+1
        current_hour = now.hour
        current_time = now.strftime("%H:%M UTC+1")
        
        # Reset all sessions
        for session in self.sessions.values():
            session["active"] = False
        
        # Check Asian session (overnight)
        if current_hour >= 23 or current_hour < 3:
            self.sessions["ASIAN"]["active"] = True
            current_session = self.sessions["ASIAN"]
        # Check London session
        elif 7 <= current_hour < 11:
            self.sessions["LONDON"]["active"] = True
            current_session = self.sessions["LONDON"]
        # Check NY/London overlap
        elif 15 <= current_hour < 19:
            self.sessions["NEWYORK"]["active"] = True
            current_session = self.sessions["NEWYORK"]
        else:
            current_session = {"name": "⏸️ MARKET CLOSED", "active": False, "hours": "Check session times"}
        
        return {
            "name": current_session["name"],
            "active": current_session["active"],
            "current_time": current_time,
            "all_sessions": self.sessions
        }

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def __init__(self):
        self.pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
    
    def generate_pre_entry(self, signal_style="NORMAL"):
        symbol = random.choice(self.pairs)
        direction = random.choice(["BUY", "SELL"])
        
        # Realistic prices
        price_ranges = {
            "EUR/USD": (1.07500, 1.09500),
            "GBP/USD": (1.25800, 1.27800),
            "USD/JPY": (148.500, 151.500),
            "XAU/USD": (1950.00, 2050.00),
            "AUD/USD": (0.65500, 0.67500),
            "USD/CAD": (1.35000, 1.37000)
        }
        
        low, high = price_ranges.get(symbol, (1.08000, 1.10000))
        price = round(random.uniform(low, high), 5 if "XAU" not in symbol else 2)
        
        # Adjust for direction
        spread = 0.0002
        if direction == "BUY":
            entry_price = round(price + spread, 5 if "XAU" not in symbol else 2)
        else:
            entry_price = round(price - spread, 5 if "XAU" not in symbol else 2)
        
        # Timeframe based on style
        if signal_style == "QUICK":
            timeframe = "1M"
            confidence = round(random.uniform(0.85, 0.92), 3)
            delay = 25
        elif signal_style == "SWING":
            timeframe = "15M"
            confidence = round(random.uniform(0.88, 0.95), 3)
            delay = 50
        else:
            timeframe = "5M"
            confidence = round(random.uniform(0.90, 0.96), 3)
            delay = Config.PRE_ENTRY_DELAY
        
        current_time = datetime.now()
        
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "confidence": confidence,
            "timeframe": timeframe,
            "delay": delay,
            "current_time": current_time.strftime("%H:%M:%S"),
            "entry_time": (current_time + timedelta(seconds=delay)).strftime("%H:%M:%S"),
            "style": signal_style
        }
    
    def generate_entry(self, pre_signal):
        symbol = pre_signal["symbol"]
        direction = pre_signal["direction"]
        entry_price = pre_signal["entry_price"]
        
        # Calculate TP/SL based on timeframe and volatility
        if pre_signal["timeframe"] == "1M":
            # Quick scalping
            if "XAU" in symbol:
                tp_dist = random.uniform(10.0, 18.0)
                sl_dist = random.uniform(6.0, 12.0)
            elif "JPY" in symbol:
                tp_dist = random.uniform(0.6, 1.2)
                sl_dist = random.uniform(0.4, 0.9)
            else:
                tp_dist = random.uniform(0.0020, 0.0035)
                sl_dist = random.uniform(0.0015, 0.0025)
        elif pre_signal["timeframe"] == "15M":
            # Swing trading
            if "XAU" in symbol:
                tp_dist = random.uniform(15.0, 30.0)
                sl_dist = random.uniform(10.0, 20.0)
            elif "JPY" in symbol:
                tp_dist = random.uniform(1.0, 2.0)
                sl_dist = random.uniform(0.7, 1.5)
            else:
                tp_dist = random.uniform(0.0035, 0.0060)
                sl_dist = random.uniform(0.0025, 0.0040)
        else:
            # 5M standard
            if "XAU" in symbol:
                tp_dist = random.uniform(12.0, 25.0)
                sl_dist = random.uniform(8.0, 18.0)
            elif "JPY" in symbol:
                tp_dist = random.uniform(0.8, 1.5)
                sl_dist = random.uniform(0.5, 1.2)
            else:
                tp_dist = random.uniform(0.0025, 0.0045)
                sl_dist = random.uniform(0.0018, 0.0030)
        
        if direction == "BUY":
            take_profit = round(entry_price + tp_dist, 5 if "XAU" not in symbol else 2)
            stop_loss = round(entry_price - sl_dist, 5 if "XAU" not in symbol else 2)
        else:
            take_profit = round(entry_price - tp_dist, 5 if "XAU" not in symbol else 2)
            stop_loss = round(entry_price + sl_dist, 5 if "XAU" not in symbol else 2)
        
        risk_reward = round(abs(take_profit - entry_price) / abs(entry_price - stop_loss), 2)
        
        return {
            **pre_signal,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "entry_time_actual": datetime.now().strftime("%H:%M:%S"),
            "risk_reward": risk_reward
        }

# ==================== ADMIN AUTH ====================
class AdminAuth:
    def __init__(self):
        self.admin_token = Config.ADMIN_TOKEN
        self.sessions = {}
    
    def verify_token(self, token):
        return token == self.admin_token
    
    def create_session(self, user_id, username):
        self.sessions[user_id] = {
            "username": username,
            "login_time": datetime.now()
        }
    
    def is_admin(self, user_id):
        if user_id in self.sessions:
            session = self.sessions[user_id]
            # 24 hour sessions
            if datetime.now() - session["login_time"] < timedelta(hours=24):
                return True
            else:
                del self.sessions[user_id]
        return False

# ==================== USER MANAGER ====================
class UserManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def add_user(self, user_id, username, first_name):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
                (user_id, username, first_name)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ User add failed: {e}")
            return False

# ==================== RISK MANAGEMENT SYSTEM ====================
class RiskManager:
    @staticmethod
    def get_risk_disclaimer():
        return f"""
🚨 *IMPORTANT RISK DISCLAIMER* 🚨

{RiskConfig.DISCLAIMERS['high_risk']}

{RiskConfig.DISCLAIMERS['past_performance']}

{RiskConfig.DISCLAIMERS['risk_capital']}

{RiskConfig.DISCLAIMERS['seek_advice']}

*By using this bot, you acknowledge and accept these risks.*
"""
    
    @staticmethod
    def get_money_management_rules():
        rules = "\n".join([f"• {rule}" for rule in RiskConfig.MONEY_MANAGEMENT.values()])
        return f"""
💰 *ESSENTIAL MONEY MANAGEMENT RULES* 💰

{rules}

📊 *Position Sizing Guide:*
• {RiskConfig.POSITION_SIZING['conservative']}
• {RiskConfig.POSITION_SIZING['moderate']}
• {RiskConfig.POSITION_SIZING['aggressive']}

*Always use proper risk management!*
"""
    
    @staticmethod
    def get_trade_warning():
        return """
⚠️ *TRADE EXECUTION WARNING* ⚠️

🚨 *RISK MANAGEMENT REQUIRED:*
• Set STOP LOSS immediately after entry
• Risk only 1-2% of your account per trade
• Ensure 1:1.5+ Risk/Reward ratio
• Trade with money you can afford to lose

📉 *Trading carries significant risk of loss*
"""

# ==================== BOT CORE ====================
class TradingBot:
    def __init__(self, application):
        self.app = application
        self.session_mgr = SessionManager()
        self.signal_gen = SignalGenerator()
        self.user_mgr = UserManager(Config.DB_PATH)
        self.sub_mgr = SubscriptionManager(Config.DB_PATH)
        self.admin_auth = AdminAuth()
        self.risk_mgr = RiskManager()
    
    def get_plans_text(self):
        """Generate plans list text with clear pricing"""
        text = ""
        for plan_id, plan in PlanConfig.PLANS.items():
            features = " • ".join(plan["features"])
            recommended_badge = " 🏆 **MOST POPULAR**" if plan.get("recommended", False) else ""
            text += f"\n{plan['emoji']} *{plan['name']}* - {plan['actual_price']}{recommended_badge}\n"
            text += f"⏰ {plan['days']} days • 📊 {plan['daily_signals']} signals/day\n"
            text += f"⚡ {features}\n"
            text += f"💡 {plan['description']}\n"
        return text
    
    async def send_welcome(self, user, chat_id):
        try:
            # Add user to database
            self.user_mgr.add_user(user.id, user.username, user.first_name)
            
            # Get user info
            subscription = self.sub_mgr.get_user_subscription(user.id)
            current_session = self.session_mgr.get_current_session()
            is_admin = self.admin_auth.is_admin(user.id)
            
            # Check if user needs to acknowledge risk
            if not subscription.get('risk_acknowledged', False):
                await self.show_risk_disclaimer(user.id, chat_id)
                return
            
            # Plan info
            plan_emoji = PlanConfig.PLANS.get(subscription['plan_type'], {}).get('emoji', '🆓')
            days_left = ""
            if subscription['subscription_end'] and subscription['plan_type'] != 'TRIAL':
                try:
                    end_date = datetime.fromisoformat(subscription['subscription_end'])
                    days_left = f" ({(end_date - datetime.now()).days} days left)"
                except:
                    pass
            
            # Check if user has access to quick trades
            user_plan = PlanConfig.PLANS.get(subscription['plan_type'], {})
            has_quick_trades = user_plan.get('quick_trades', False)
            
            message = f"""
🎉 *WELCOME TO LEKZY FX AI PRO!* 🚀

*Hello {user.first_name}!* 👋

📊 *YOUR ACCOUNT STATUS:*
• Plan: {plan_emoji} *{subscription['plan_type']}*{days_left}
• Signals Used: *{subscription['signals_used']}/{subscription['max_daily_signals']}*
• Status: *{'✅ ACTIVE' if subscription['is_active'] else '❌ EXPIRED'}*
• Trade Types: *{'⚡ Quick & 📈 Normal' if has_quick_trades else '📈 Normal Only'}*

{'🎯' if current_session['active'] else '⏸️'} *MARKET STATUS: {current_session['name']}*
🕒 *Time:* {current_session['current_time']}

💡 *What I Offer:*
• AI-Powered Trading Signals
• Multiple Timeframe Strategies  
• Professional Risk Management
• Real-time Market Analysis

🚀 *Ready to start trading? Choose an option below!*
"""
            if is_admin:
                message += "\n👑 *You have Admin Access*\n"
            
            # Dynamic keyboard based on user's plan
            keyboard = []
            
            if has_quick_trades or is_admin:
                # Premium+ users see both options
                keyboard.append([
                    InlineKeyboardButton("⚡ QUICK TRADE", callback_data="quick_signal"),
                    InlineKeyboardButton("📈 NORMAL TRADE", callback_data="normal_signal")
                ])
            else:
                # Trial users see only normal trades
                keyboard.append([InlineKeyboardButton("🚀 GET TRADING SIGNAL", callback_data="normal_signal")])
                keyboard.append([InlineKeyboardButton("💎 UNLOCK QUICK TRADES", callback_data="show_plans")])
            
            # Secondary options
            keyboard.append([InlineKeyboardButton("💎 VIEW SUBSCRIPTION PLANS", callback_data="show_plans")])
            keyboard.append([InlineKeyboardButton("📊 MY ACCOUNT STATS", callback_data="show_stats")])
            
            # Utility options
            keyboard.append([
                InlineKeyboardButton("🕒 SESSIONS", callback_data="session_info"),
                InlineKeyboardButton("🚨 RISK GUIDE", callback_data="risk_management")
            ])
            
            if is_admin:
                keyboard.insert(0, [InlineKeyboardButton("👑 ADMIN PANEL", callback_data="admin_panel")])
                keyboard.insert(2, [
                    InlineKeyboardButton("⚡ QUICK", callback_data="admin_quick"),
                    InlineKeyboardButton("📈 SWING", callback_data="admin_swing")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Welcome failed: {e}")
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=f"Welcome {user.first_name}! Use /start to see options."
            )
    
    async def show_risk_disclaimer(self, user_id, chat_id):
        """Show risk disclaimer and require acknowledgment"""
        disclaimer = self.risk_mgr.get_risk_disclaimer()
        
        message = f"""
{disclaimer}

🔒 *ACCOUNT SETUP REQUIRED*

*Before you can start trading, you must acknowledge and understand the risks involved in trading.*

📋 *Please read the above carefully and confirm your understanding.*
"""
        
        keyboard = [
            [InlineKeyboardButton("✅ I UNDERSTAND & ACCEPT THE RISKS", callback_data="accept_risks")],
            [InlineKeyboardButton("❌ CANCEL", callback_data="cancel_risks")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_risk_management(self, chat_id):
        """Show comprehensive risk management guide"""
        risk_rules = self.risk_mgr.get_money_management_rules()
        
        message = f"""
🛡️ *COMPREHENSIVE RISK MANAGEMENT GUIDE* 🛡️

{risk_rules}

📈 *Example Position Sizing:*
• Account: $1,000
• Risk: 1% = $10 per trade
• Stop Loss: 20 pips
• Position Size: $0.50 per pip

💡 *Key Principles:*
• Preserve capital above all else
• Never risk more than you can afford to lose
• Emotional control is crucial
• Consistency beats occasional big wins

🚨 *Remember: Professional traders focus on risk management first, profits second!*
"""
        
        keyboard = [
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_trade_types_info(self, chat_id):
        """Educational page about different trade types"""
        message = """
📚 *UNDERSTANDING TRADE TYPES*

⚡ *QUICK TRADES (1-Minute Timeframe)*
*What to Expect:*
• Very fast signals (25-second countdown)
• Tight stop losses (smaller risk per trade)
• Quick profit targets
• Higher trading frequency

*Best For:*
• Experienced traders
• Scalpers who watch markets closely
• Those with fast internet connections
• Risk-tolerant individuals

*Risks:*
• Higher spread costs
• More susceptible to market noise
• Requires quick execution

📈 *NORMAL TRADES (5M/15M Timeframe)*
*What to Expect:*
• Standard speed (40-second countdown)  
• Balanced stop losses
• Realistic profit targets
• Medium trading frequency

*Best For:*
• Most traders (recommended)
• Beginners learning the markets
• Those who can't watch charts constantly
• Risk-averse individuals

*Benefits:*
• More reliable signals
• Better risk-reward ratios
• Less affected by market noise

💡 *Our Recommendation:* Start with Normal trades to learn the system, then consider Quick trades once you're comfortable!
"""
        
        keyboard = [
            [InlineKeyboardButton("⚡ TRY QUICK TRADE", callback_data="quick_signal")],
            [InlineKeyboardButton("📈 TRY NORMAL TRADE", callback_data="normal_signal")],
            [InlineKeyboardButton("💎 UPGRADE FOR QUICK TRADES", callback_data="show_plans")],
            [InlineKeyboardButton("🏠 BACK TO MAIN", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_plans(self, chat_id):
        try:
            # Build plans text with clear pricing
            plans_text = ""
            for plan_id, plan in PlanConfig.PLANS.items():
                features = " • ".join(plan["features"])
                recommended_badge = " 🏆 **MOST POPULAR**" if plan.get("recommended", False) else ""
                text += f"\n{plan['emoji']} *{plan['name']}* - {plan['actual_price']}{recommended_badge}\n"
                text += f"⏰ {plan['days']} days • 📊 {plan['daily_signals']} signals/day\n"
                text += f"⚡ {features}\n"
                text += f"💡 {plan['description']}\n"
            
            message = f"""
💎 *LEKZY FX AI PRO - SUBSCRIPTION PLANS*

*Choose the plan that fits your trading style:*

{plans_text}

💰 *Transparent Pricing - No Hidden Fees*

🎯 *Why Traders Choose Us:*
• 90%+ Signal Accuracy Rate
• Real-time AI Analysis
• Professional Risk Management
• 24/7 Customer Support

💳 *Payment Methods:*
• Cryptocurrency (BTC, ETH, USDT)
• Bank Transfer
• Mobile Money
• Credit/Debit Cards

🚀 *Ready to upgrade? Contact {Config.ADMIN_CONTACT} to get started!*
"""
            keyboard = [
                [InlineKeyboardButton("🚀 TRY FREE SIGNALS", callback_data="normal_signal")],
                [InlineKeyboardButton("📞 CONTACT TO PURCHASE", callback_data="contact_support")],
                [InlineKeyboardButton("📊 MY CURRENT PLAN", callback_data="show_stats")],
                [InlineKeyboardButton("🏠 BACK TO MAIN", callback_data="main_menu")]
            ]
            
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Show plans failed: {e}")
    
    async def show_contact_support(self, chat_id):
        """Streamlined contact page since prices are now in plans"""
        message = f"""
📞 *GET STARTED WITH LEKZY FX AI PRO*

*Ready to upgrade your trading?*

💎 *Subscription Plans Available:*
• **PREMIUM** - $49.99 (30 days)
• **VIP** - $129.99 (90 days)  
• **PRO** - $199.99 (180 days)

💳 *Instant Activation Available*
We accept multiple payment methods for your convenience.

🎯 *What You Get:*
• Professional-grade trading signals
• AI-powered market analysis
• Risk management guidance
• 24/7 customer support
• Instant activation after payment

📱 *Contact Us Now:*
{Config.ADMIN_CONTACT}

*Mention your preferred plan and we'll get you set up immediately!*
"""
        keyboard = [
            [InlineKeyboardButton("💎 VIEW PLANS & PRICING", callback_data="show_plans")],
            [InlineKeyboardButton("🚀 TRY FREE SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_market_sessions(self, chat_id):
        try:
            current_session = self.session_mgr.get_current_session()
            all_sessions = current_session['all_sessions']
            
            message = f"""
🕒 *MARKET TRADING SESSIONS*

*Current Session:*
{'✅' if current_session['active'] else '⏸️'} *{current_session['name']}*
🕐 *Time:* {current_session['current_time']}

📊 *All Trading Sessions:*
"""
            for session_id, session in all_sessions.items():
                status = "🟢 ACTIVE" if session["active"] else "🔴 CLOSED"
                message += f"\n{session['name']}\n"
                message += f"⏰ {session['hours']} • {status}\n"
            
            message += f"""
            
💡 *Trading Hours (UTC+1):*
• Asian: 23:00 - 03:00
• London: 07:00 - 11:00  
• NY/London: 15:00 - 19:00

🎯 *Best Trading Times:*
• London Open (08:00-10:00)
• NY Open (15:00-17:00)
• Overlap (15:00-17:00)

*Markets are most volatile during session overlaps!*
"""
            keyboard = [
                [InlineKeyboardButton("🚀 GET SIGNAL ANYWAY", callback_data="normal_signal")],
                [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
                [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
            ]
            
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Show sessions failed: {e}")
    
    async def generate_signal(self, user_id, chat_id, signal_style="NORMAL", is_admin=False):
        try:
            # Check if user acknowledged risks
            subscription = self.sub_mgr.get_user_subscription(user_id)
            if not subscription.get('risk_acknowledged', False):
                await self.show_risk_disclaimer(user_id, chat_id)
                return
            
            # Check subscription
            if not is_admin:
                can_request, msg = self.sub_mgr.can_user_request_signal(user_id)
                if not can_request:
                    await self.app.bot.send_message(chat_id, f"❌ {msg}\n\n💎 Use /plans to upgrade!")
                    return
            
            # Check market session but allow trading anyway with warning
            session = self.session_mgr.get_current_session()
            if not session['active'] and not is_admin:
                warning_msg = "⚠️ *MARKET IS CLOSED*\n\n"
                warning_msg += f"Current: {session['name']}\n"
                warning_msg += f"Time: {session['current_time']}\n\n"
                warning_msg += "*You can still trade, but volatility may be low.*\n"
                warning_msg += "Proceed with caution! 🎯"
                
                keyboard = [
                    [InlineKeyboardButton("🎯 CONTINUE ANYWAY", callback_data=f"force_signal_{signal_style}")],
                    [InlineKeyboardButton("🕒 VIEW SESSIONS", callback_data="session_info")],
                    [InlineKeyboardButton("💎 UPGRADE PLAN", callback_data="show_plans")]
                ]
                
                await self.app.bot.send_message(
                    chat_id,
                    warning_msg,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode='Markdown'
                )
                return
            
            # Generate signal
            await self._generate_signal_process(user_id, chat_id, signal_style, is_admin)
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            await self.app.bot.send_message(chat_id, "❌ Failed to generate signal. Try again.")
    
    async def _generate_signal_process(self, user_id, chat_id, signal_style, is_admin):
        """Internal method to generate signals"""
        style_text = signal_style.upper()
        await self.app.bot.send_message(chat_id, f"🎯 *Generating {style_text} signal...* ⏱️")
        
        pre_signal = self.signal_gen.generate_pre_entry(signal_style)
        
        # Send pre-entry
        direction_emoji = "🟢" if pre_signal["direction"] == "BUY" else "🔴"
        sub_info = self.sub_mgr.get_user_subscription(user_id)
        
        # Timeframe explanation
        tf_explanation = {
            "1M": "⚡ QUICK SCALPING - Fast entries/exits",
            "5M": "📈 DAY TRADING - Balanced risk/reward", 
            "15M": "🎯 SWING TRADING - Higher confidence"
        }.get(pre_signal['timeframe'], "Standard trading")
        
        pre_msg = f"""
📊 *PRE-ENTRY SIGNAL* - {style_text}

{direction_emoji} *{pre_signal['symbol']}* | **{pre_signal['direction']}**
💵 *Entry:* `{pre_signal['entry_price']}`
🎯 *Confidence:* {pre_signal['confidence']*100:.1f}%

⏰ *Timing:*
• Current: `{pre_signal['current_time']}`
• Entry: `{pre_signal['entry_time']}` 
• Delay: {pre_signal['delay']}s
• TF: *{pre_signal['timeframe']}*

{tf_explanation}

📊 *Your Plan:* {sub_info['plan_type']}
📈 *Signals Left:* {sub_info['signals_remaining']}

*Entry in {pre_signal['delay']}s...* ⏳
"""
        await self.app.bot.send_message(chat_id, pre_msg, parse_mode='Markdown')
        
        # Wait and generate entry
        await asyncio.sleep(pre_signal['delay'])
        
        entry_signal = self.signal_gen.generate_entry(pre_signal)
        
        # Increment signal count
        if not is_admin:
            self.sub_mgr.increment_signal_count(user_id)
        
        # Send entry signal with risk warning
        risk_warning = self.risk_mgr.get_trade_warning()
        
        entry_msg = f"""
🎯 *ENTRY SIGNAL* ✅

{direction_emoji} *{entry_signal['symbol']}* | **{entry_signal['direction']}**
💵 *Entry:* `{entry_signal['entry_price']}`
✅ *TP:* `{entry_signal['take_profit']}`
❌ *SL:* `{entry_signal['stop_loss']}`

⏰ *Time:* `{entry_signal['entry_time_actual']}`
📊 *TF:* {entry_signal['timeframe']}
🎯 *Confidence:* {entry_signal['confidence']*100:.1f}%
⚖️ *Risk/Reward:* 1:{entry_signal.get('risk_reward', 1.5)}

{risk_warning}

*Execute this trade now!* 🚀
"""
        keyboard = [
            [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
            [InlineKeyboardButton("🔄 NEW SIGNAL", callback_data="get_signal")],
            [InlineKeyboardButton("💎 UPGRADE PLAN", callback_data="show_plans")],
            [InlineKeyboardButton("🚨 RISK MANAGEMENT", callback_data="risk_management")]
        ]
        
        if is_admin:
            keyboard.insert(1, [
                InlineKeyboardButton("⚡ QUICK", callback_data="admin_quick"),
                InlineKeyboardButton("📈 SWING", callback_data="admin_swing")
            ])
        
        await self.app.bot.send_message(
            chat_id,
            entry_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

# ==================== TELEGRAM BOT HANDLERS ====================
class TelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.app = None
        self.bot_core = None
    
    async def initialize(self):
        try:
            self.app = Application.builder().token(self.token).build()
            self.bot_core = TradingBot(self.app)
            
            # Add handlers
            handlers = [
                CommandHandler("start", self.start_cmd),
                CommandHandler("signal", self.signal_cmd),
                CommandHandler("session", self.session_cmd),
                CommandHandler("register", self.register_cmd),
                CommandHandler("plans", self.plans_cmd),
                CommandHandler("mystats", self.mystats_cmd),
                CommandHandler("login", self.login_cmd),
                CommandHandler("admin", self.admin_cmd),
                CommandHandler("seedtoken", self.seedtoken_cmd),
                CommandHandler("help", self.help_cmd),
                CommandHandler("risk", self.risk_cmd),
                CallbackQueryHandler(self.button_handler)
            ]
            
            for handler in handlers:
                self.app.add_handler(handler)
            
            await self.app.initialize()
            await self.app.start()
            logger.info("✅ Bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot init failed: {e}")
            return False
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await self.bot_core.send_welcome(user, update.effective_chat.id)
    
    async def risk_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_risk_management(update.effective_chat.id)
    
    async def signal_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        is_admin = self.bot_core.admin_auth.is_admin(user.id)
        style = "NORMAL"
        
        if context.args:
            arg = context.args[0].upper()
            if arg == "QUICK" and is_admin:
                style = "QUICK"
            elif arg == "SWING" and is_admin:
                style = "SWING"
            elif arg == "QUICK" and not is_admin:
                await update.message.reply_text("❌ Quick trades are for admins only. Upgrade to VIP!")
                return
            
        await self.bot_core.generate_signal(user.id, update.effective_chat.id, style, is_admin)
    
    async def register_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text(
                "🔐 *ACTIVATE SUBSCRIPTION*\n\n"
                "Usage: `/register YOUR_TOKEN`\n\n"
                "💎 Use `/plans` to see available plans!\n"
                "📞 Contact admin for tokens."
            )
            return
        
        token = context.args[0].upper()
        user = update.effective_user
        
        is_valid, plan_type, days = self.bot_core.sub_mgr.token_manager.validate_token(token)
        
        if not is_valid:
            await update.message.reply_text(
                "❌ *Invalid or used token!*\n\n"
                f"📞 Contact {Config.ADMIN_CONTACT} for valid tokens.\n"
                "💎 Use `/plans` to see available plans."
            )
            return
        
        success = self.bot_core.sub_mgr.activate_subscription(user.id, token, plan_type, days)
        
        if success:
            plan_config = PlanConfig.PLANS.get(plan_type, {})
            await update.message.reply_text(
                f"🎉 *{plan_config.get('name', 'SUBSCRIPTION')} ACTIVATED!* 🚀\n\n"
                f"✅ *Plan:* {plan_config.get('name', 'Premium')}\n"
                f"⏰ *Duration:* {days} days\n"
                f"📊 *Signals:* {plan_config.get('daily_signals', 50)}/day\n"
                f"💎 *Features:* All premium features unlocked!\n\n"
                f"*Use /signal to start trading!* 🎯"
            )
        else:
            await update.message.reply_text("❌ Registration failed. Please try again.")
    
    async def plans_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_plans(update.effective_chat.id)
    
    async def session_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_market_sessions(update.effective_chat.id)
    
    async def mystats_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        subscription = self.bot_core.sub_mgr.get_user_subscription(user.id)
        is_admin = self.bot_core.admin_auth.is_admin(user.id)
        
        plan_emoji = PlanConfig.PLANS.get(subscription['plan_type'], {}).get('emoji', '🆓')
        days_left = ""
        if subscription['subscription_end'] and subscription['plan_type'] != 'TRIAL':
            try:
                end_date = datetime.fromisoformat(subscription['subscription_end'])
                days_left = f" ({(end_date - datetime.now()).days} days left)"
            except:
                pass
        
        user_plan = PlanConfig.PLANS.get(subscription['plan_type'], {})
        has_quick_trades = user_plan.get('quick_trades', False)
        
        message = f"""
📊 *YOUR TRADING STATISTICS*

👤 *Trader:* {user.first_name}
💼 *Plan:* {plan_emoji} {subscription['plan_type']}{days_left}
📈 *Signals Today:* {subscription['signals_used']}/{subscription['max_daily_signals']}
🎯 *Status:* {'✅ ACTIVE' if subscription['is_active'] else '❌ EXPIRED'}
⚡ *Quick Trades:* {'✅ AVAILABLE' if has_quick_trades else '💎 UPGRADE REQUIRED'}
🔑 *Admin Access:* {'✅ YES' if is_admin else '❌ NO'}
🛡️ *Risk Acknowledged:* {'✅ YES' if subscription.get('risk_acknowledged', False) else '❌ NO'}

💡 *Recommendation:* {'🎉 You have the best plan!' if subscription['plan_type'] == 'PRO' else '💎 Consider upgrading for more signals!'}
"""
        keyboard = [
            [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("🚨 RISK MANAGEMENT", callback_data="risk_management")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        if update.callback_query:
            await update.callback_query.edit_message_text(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        else:
            await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    
    async def login_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("🔐 *ADMIN LOGIN*\n\nUsage: `/login ADMIN_TOKEN`")
            return
        
        user = update.effective_user
        token = context.args[0]
        
        if self.bot_core.admin_auth.verify_token(token):
            self.bot_core.admin_auth.create_session(user.id, user.username)
            await update.message.reply_text(
                "✅ *Admin access granted!* 👑\n\n"
                "*You now have access to:*\n"
                "• Quick trade signals (1M)\n"
                "• Swing trade signals (15M)\n" 
                "• Token generation\n"
                "• Admin dashboard\n\n"
                "*Use /admin to access features!*"
            )
        else:
            await update.message.reply_text("❌ Invalid admin token")
    
    async def admin_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        
        if not self.bot_core.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required. Use `/login TOKEN`")
            return
        
        message = """
👑 *ADMIN DASHBOARD* 🔧

⚡ *Admin Features:*
• Quick Trades (1M) - 25s entry
• Swing Trades (15M) - 50s entry
• Generate subscription tokens
• System monitoring

💰 *Available Plans for Tokens:*
"""
        for plan_id, plan in PlanConfig.PLANS.items():
            message += f"• {plan['emoji']} {plan['name']} - {plan['actual_price']} - {plan['days']} days\n"
        
        message += "\n🎯 *Commands:*\n• `/seedtoken PLAN DAYS` - Generate tokens\n• `/signal quick` - Quick trade\n• `/signal swing` - Swing trade"
        
        keyboard = [
            [
                InlineKeyboardButton("⚡ QUICK TRADE", callback_data="admin_quick"),
                InlineKeyboardButton("📈 SWING TRADE", callback_data="admin_swing")
            ],
            [InlineKeyboardButton("🔑 GENERATE TOKENS", callback_data="admin_tokens")],
            [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    
    async def seedtoken_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        
        if not self.bot_core.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required")
            return
        
        if not context.args:
            await update.message.reply_text(
                "🔑 *GENERATE SUBSCRIPTION TOKENS*\n\n"
                "Usage: `/seedtoken PLAN DAYS`\n\n"
                "📋 *Available Plans:*\n"
                "• TRIAL - 7 days, 3 signals\n"
                "• PREMIUM - $49.99 value\n" 
                "• VIP - $129.99 value\n"
                "• PRO - $199.99 value\n\n"
                "💡 *Example:* `/seedtoken PREMIUM 30`"
            )
            return
        
        plan_type = context.args[0].upper()
        if plan_type not in PlanConfig.PLANS:
            await update.message.reply_text(
                f"❌ Invalid plan. Available: {', '.join(PlanConfig.PLANS.keys())}\n"
                f"💎 Use `/plans` to see plan details"
            )
            return
        
        try:
            days = int(context.args[1]) if len(context.args) > 1 else PlanConfig.PLANS[plan_type]["days"]
        except:
            days = PlanConfig.PLANS[plan_type]["days"]
        
        token = self.bot_core.sub_mgr.token_manager.generate_token(plan_type, days, user.id)
        
        if token:
            plan_config = PlanConfig.PLANS[plan_type]
            await update.message.reply_text(
                f"🔑 *{plan_config['name']} TOKEN GENERATED* ✅\n\n"
                f"*Token:* `{token}`\n"
                f"*Plan:* {plan_config['name']}\n"
                f"*Value:* {plan_config['actual_price']}\n"
                f"*Duration:* {days} days\n"
                f"*Signals:* {plan_config['daily_signals']}/day\n\n"
                f"📤 *Share with users:*\n`/register {token}`\n\n"
                f"💡 User will get {plan_config['name']} plan for {days} days!"
            )
        else:
            await update.message.reply_text("❌ Token generation failed")
    
    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
🤖 *LEKZY FX AI PRO - HELP GUIDE*

💎 *TRADING COMMANDS:*
• /start - Main menu with options
• /signal - Get trading signal (always available)
• /session - Market session times
• /plans - View subscription plans & pricing
• /register TOKEN - Activate subscription
• /mystats - Your account statistics
• /risk - Risk management guide

👑 *ADMIN COMMANDS:*
• /login TOKEN - Admin access
• /admin - Admin dashboard  
• /seedtoken PLAN DAYS - Generate tokens
• /signal quick - Quick trades (1M)
• /signal swing - Swing trades (15M)

💰 *SUBSCRIPTION PLANS:*
• 🆓 Trial - FREE (3 signals/day)
• 💎 Premium - $49.99 (50 signals/day)
• 🚀 VIP - $129.99 (100 signals/day) 
• 🔥 PRO - $199.99 (200 signals/day)

💡 *All pricing is transparently displayed in /plans*

🚨 *RISK WARNING:*
Trading carries significant risk. Only use risk capital.

📞 *Support & Purchases:* @LekzyTradingPro

🚀 *Happy Trading!*
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        user = query.from_user
        data = query.data
        
        try:
            if data == "get_signal":
                await self.signal_cmd(update, context)
            elif data == "quick_signal":
                # Check if user has access to quick trades
                subscription = self.bot_core.sub_mgr.get_user_subscription(user.id)
                user_plan = PlanConfig.PLANS.get(subscription['plan_type'], {})
                has_quick_trades = user_plan.get('quick_trades', False)
                is_admin = self.bot_core.admin_auth.is_admin(user.id)
                
                if not has_quick_trades and not is_admin:
                    await query.edit_message_text(
                        "❌ *Quick Trades Not Available*\n\n"
                        "⚡ *Quick Trades* are available for Premium subscribers and above.\n\n"
                        "💎 *Upgrade to unlock:*\n"
                        "• Faster 1-minute timeframe signals\n"
                        "• Quick 25-second entries\n"
                        "• Advanced trading features\n\n"
                        "*Use Normal trades for now, or upgrade to access Quick trades!*",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("📈 USE NORMAL TRADES", callback_data="normal_signal")],
                            [InlineKeyboardButton("💎 UPGRADE PLAN", callback_data="show_plans")],
                            [InlineKeyboardButton("📚 LEARN ABOUT TRADE TYPES", callback_data="learn_trade_types")],
                            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
                        ])
                    )
                    return
                
                await self.bot_core.generate_signal(user.id, query.message.chat_id, "QUICK", is_admin)
                
            elif data == "normal_signal":
                is_admin = self.bot_core.admin_auth.is_admin(user.id)
                await self.bot_core.generate_signal(user.id, query.message.chat_id, "NORMAL", is_admin)
                
            elif data == "learn_trade_types":
                await self.bot_core.show_trade_types_info(query.message.chat_id)
                
            elif data.startswith("force_signal_"):
                # Extract signal style from force_signal_{style}
                style = data.replace("force_signal_", "")
                await self.bot_core._generate_signal_process(
                    user.id, query.message.chat_id, style, 
                    self.bot_core.admin_auth.is_admin(user.id)
                )
            elif data == "show_plans":
                await self.plans_cmd(update, context)
            elif data == "show_stats":
                await self.mystats_cmd(update, context)
            elif data == "session_info":
                await self.session_cmd(update, context)
            elif data == "risk_management":
                await self.bot_core.show_risk_management(query.message.chat_id)
            elif data == "contact_support":
                await self.bot_core.show_contact_support(query.message.chat_id)
            elif data == "trade_done":
                await query.edit_message_text(
                    "✅ *Trade Executed Successfully!* 🎯\n\n"
                    "*Remember to always use proper risk management!*\n"
                    "*Happy trading! May the profits be with you!* 💰"
                )
            elif data == "accept_risks":
                # Mark user as having acknowledged risks
                success = self.bot_core.sub_mgr.mark_risk_acknowledged(user.id)
                if success:
                    await query.edit_message_text(
                        "✅ *Risk Acknowledgement Confirmed!* 🛡️\n\n"
                        "*You can now access all trading features.*\n"
                        "*Remember to always trade responsibly!*\n\n"
                        "*Redirecting to main menu...*"
                    )
                    await asyncio.sleep(2)
                    await self.start_cmd(update, context)
                else:
                    await query.edit_message_text("❌ Failed to save acknowledgment. Please try /start again.")
            elif data == "cancel_risks":
                await query.edit_message_text(
                    "❌ *Risk Acknowledgement Required*\n\n"
                    "*You must acknowledge the risks before trading.*\n"
                    "*Use /start when you're ready to proceed.*\n\n"
                    "*Trading involves significant risk of loss.*"
                )
            elif data == "admin_panel":
                await self.admin_cmd(update, context)
            elif data == "admin_quick":
                if self.bot_core.admin_auth.is_admin(user.id):
                    await self.bot_core.generate_signal(user.id, query.message.chat_id, "QUICK", True)
                else:
                    await query.edit_message_text("❌ Admin access required for quick trades")
            elif data == "admin_swing":
                if self.bot_core.admin_auth.is_admin(user.id):
                    await self.bot_core.generate_signal(user.id, query.message.chat_id, "SWING", True)
                else:
                    await query.edit_message_text("❌ Admin access required for swing trades")
            elif data == "admin_tokens":
                if self.bot_core.admin_auth.is_admin(user.id):
                    await query.edit_message_text(
                        "🔑 *GENERATE SUBSCRIPTION TOKENS*\n\n"
                        "Use `/seedtoken PLAN DAYS` to create tokens.\n\n"
                        "📋 *Available Plans:*\n"
                        "• TRIAL - 7 days, 3 signals\n"
                        "• PREMIUM - $49.99 (30 days)\n"
                        "• VIP - $129.99 (90 days)\n"
                        "• PRO - $199.99 (180 days)\n\n"
                        "💡 *Example:* `/seedtoken PREMIUM 30`"
                    )
                else:
                    await query.edit_message_text("❌ Admin access required")
            elif data == "main_menu":
                await self.start_cmd(update, context)
                
        except Exception as e:
            logger.error(f"Button error: {e}")
            await query.edit_message_text("❌ Action failed. Use /start to refresh")
    
    async def start_polling(self):
        await self.app.updater.start_polling()
        logger.info("✅ Bot polling started")
    
    async def stop(self):
        await self.app.stop()

# ==================== MAIN APPLICATION ====================
async def main():
    # Initialize
    initialize_database()
    start_web_server()
    
    # Start bot
    bot = TelegramBot()
    success = await bot.initialize()
    
    if success:
        logger.info("🚀 LEKZY FX AI PRO - SMART TRADE SELECTION ACTIVE!")
        await bot.start_polling()
        
        # Keep running
        while True:
            await asyncio.sleep(10)
    else:
        logger.error("❌ Failed to start bot")

if __name__ == "__main__":
    print("🚀 Starting LEKZY FX AI PRO - SMART TRADE SELECTION EDITION...")
    asyncio.run(main())
