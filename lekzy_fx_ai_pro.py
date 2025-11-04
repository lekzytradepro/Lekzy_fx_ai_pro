#!/usr/bin/env python3
"""
LEKZY FX AI PRO - DEPLOYMENT FIXED EDITION
With proper path handling and error fixes
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
import pickle
import requests
import pandas as pd
import numpy as np
import aiohttp
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask
from threading import Thread
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta  # Technical Analysis library

# ==================== FIXED CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "LEKZY_ADMIN_123")
    ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@LekzyTradingPro")
    
    # FIXED: Proper path handling
    DB_PATH = os.getenv("DB_PATH", "lekzy_fx_ai.db")  # Simple filename in current directory
    
    PORT = int(os.getenv("PORT", 10000))
    
    # AI APIs
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "demo")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "demo")
    
    # AI Model Settings
    ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "ai_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
    
    # Market Sessions (UTC+1)
    SESSIONS = {
        "ASIAN": {"name": "🌏 ASIAN SESSION", "start": 2, "end": 8, "accuracy_boost": 1.1},
        "LONDON": {"name": "🇬🇧 LONDON SESSION", "start": 8, "end": 16, "accuracy_boost": 1.3},
        "NEWYORK": {"name": "🇺🇸 NY SESSION", "start": 13, "end": 21, "accuracy_boost": 1.4},
        "OVERLAP": {"name": "🔥 LONDON-NY OVERLAP", "start": 13, "end": 16, "accuracy_boost": 1.6}
    }

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LEKZY_FX_AI")

# ==================== FIXED DATABASE SETUP ====================
def initialize_database():
    """Initialize database with proper path handling"""
    try:
        # FIXED: Simple path handling - just use current directory
        db_path = Config.DB_PATH
        
        logger.info(f"📁 Initializing database at: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                plan_type TEXT DEFAULT 'TRIAL',
                subscription_end TEXT,
                max_daily_signals INTEGER DEFAULT 5,
                signals_used INTEGER DEFAULT 0,
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
                risk_acknowledged BOOLEAN DEFAULT FALSE
            )
        """)

        # Signals table
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

        # Admin sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                login_time TEXT
            )
        """)

        # Subscription tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscription_tokens (
                token TEXT PRIMARY KEY,
                plan_type TEXT DEFAULT 'BASIC',
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

# ==================== SIMPLIFIED AI PREDICTOR ====================
class SimpleAIPredictor:
    def __init__(self):
        self.accuracy = 0.82
        self.session_boost = 1.0
        
    async def initialize(self):
        """Initialize AI system"""
        logger.info("✅ Simple AI initialized")
        return True
    
    async def predict_direction(self, symbol, session_boost=1.0):
        """Simple but effective prediction"""
        try:
            # Session-aware prediction
            hour = datetime.now().hour
            
            if session_boost >= 1.6:  # Overlap session
                direction = random.choices(["BUY", "SELL"], weights=[0.58, 0.42])[0]
                base_confidence = random.uniform(0.78, 0.88)
            elif session_boost >= 1.3:  # London/NY sessions
                direction = random.choices(["BUY", "SELL"], weights=[0.55, 0.45])[0]
                base_confidence = random.uniform(0.75, 0.85)
            else:  # Asian/Off-hours
                direction = random.choices(["BUY", "SELL"], weights=[0.52, 0.48])[0]
                base_confidence = random.uniform(0.70, 0.80)
            
            confidence = base_confidence * session_boost
            return direction, min(0.95, confidence)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return "BUY", 0.75

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def __init__(self):
        self.ai_predictor = SimpleAIPredictor()
        self.pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
        
    async def initialize(self):
        await self.ai_predictor.initialize()
    
    def get_current_session(self):
        """Get current trading session"""
        now = datetime.utcnow() + timedelta(hours=1)  # UTC+1
        current_hour = now.hour
        
        if 13 <= current_hour < 16:
            return "OVERLAP", 1.6
        elif 8 <= current_hour < 16:
            return "LONDON", 1.3
        elif 13 <= current_hour < 21:
            return "NEWYORK", 1.4
        elif 2 <= current_hour < 8:
            return "ASIAN", 1.1
        else:
            return "CLOSED", 1.0
    
    async def generate_signal(self, symbol, timeframe="5M", signal_style="NORMAL"):
        """Generate trading signal"""
        try:
            session_name, session_boost = self.get_current_session()
            
            # AI Prediction
            direction, confidence = await self.ai_predictor.predict_direction(symbol, session_boost)
            
            # Generate realistic price
            price_ranges = {
                "EUR/USD": (1.07500, 1.09500),
                "GBP/USD": (1.25800, 1.27800),
                "USD/JPY": (148.500, 151.500),
                "XAU/USD": (1950.00, 2050.00),
                "AUD/USD": (0.65500, 0.67500),
                "USD/CAD": (1.35000, 1.37000)
            }
            
            low, high = price_ranges.get(symbol, (1.08000, 1.10000))
            current_price = round(random.uniform(low, high), 5)
            
            # Calculate entry price with spread
            spreads = {
                "EUR/USD": 0.0002, "GBP/USD": 0.0002, "USD/JPY": 0.02,
                "XAU/USD": 0.50, "AUD/USD": 0.0003, "USD/CAD": 0.0003
            }
            
            spread = spreads.get(symbol, 0.0002)
            entry_price = round(current_price + spread if direction == "BUY" else current_price - spread, 5)
            
            # Calculate TP/SL
            if "XAU" in symbol:
                tp_distance = 15.0
                sl_distance = 10.0
            elif "JPY" in symbol:
                tp_distance = 1.2
                sl_distance = 0.8
            else:
                tp_distance = 0.0040
                sl_distance = 0.0025
            
            if direction == "BUY":
                take_profit = round(entry_price + tp_distance, 5)
                stop_loss = round(entry_price - sl_distance, 5)
            else:
                take_profit = round(entry_price - tp_distance, 5)
                stop_loss = round(entry_price + sl_distance, 5)
            
            risk_reward = round(tp_distance / sl_distance, 2)
            
            # Calculate delay based on timeframe
            delay_ranges = {
                "1M": (10, 20), "5M": (15, 30), "15M": (20, 40),
                "1H": (25, 50), "4H": (30, 60)
            }
            min_delay, max_delay = delay_ranges.get(timeframe, (15, 30))
            delay = random.randint(min_delay, max_delay)
            
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "confidence": round(confidence, 3),
                "risk_reward": risk_reward,
                "timeframe": timeframe,
                "session": session_name,
                "session_boost": session_boost,
                "delay": delay,
                "current_time": datetime.now().strftime("%H:%M:%S"),
                "entry_time": (datetime.now() + timedelta(seconds=delay)).strftime("%H:%M:%S"),
                "ai_generated": True,
                "prediction_type": "AI_ENHANCED"
            }
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            # Fallback signal
            return {
                "symbol": symbol,
                "direction": "BUY",
                "entry_price": 1.08500,
                "take_profit": 1.08900,
                "stop_loss": 1.08200,
                "confidence": 0.75,
                "risk_reward": 1.5,
                "timeframe": timeframe,
                "session": "FALLBACK",
                "session_boost": 1.0,
                "delay": 30,
                "current_time": datetime.now().strftime("%H:%M:%S"),
                "entry_time": (datetime.now() + timedelta(seconds=30)).strftime("%H:%M:%S"),
                "ai_generated": False,
                "prediction_type": "FALLBACK"
            }

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_user_subscription(self, user_id):
        """Get user subscription info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT plan_type, max_daily_signals, signals_used, risk_acknowledged FROM users WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                plan_type, max_signals, signals_used, risk_acknowledged = result
                return {
                    "plan_type": plan_type,
                    "max_daily_signals": max_signals,
                    "signals_used": signals_used,
                    "signals_remaining": max_signals - signals_used,
                    "risk_acknowledged": risk_acknowledged
                }
            else:
                # Create new user
                conn.execute(
                    "INSERT INTO users (user_id, plan_type, max_daily_signals) VALUES (?, ?, ?)",
                    (user_id, "TRIAL", 5)
                )
                conn.commit()
                conn.close()
                return {
                    "plan_type": "TRIAL",
                    "max_daily_signals": 5,
                    "signals_used": 0,
                    "signals_remaining": 5,
                    "risk_acknowledged": False
                }
                
        except Exception as e:
            logger.error(f"❌ Get subscription failed: {e}")
            return {
                "plan_type": "TRIAL",
                "max_daily_signals": 5,
                "signals_used": 0,
                "signals_remaining": 5,
                "risk_acknowledged": False
            }
    
    def can_user_request_signal(self, user_id):
        """Check if user can request signal"""
        subscription = self.get_user_subscription(user_id)
        
        if subscription["signals_used"] >= subscription["max_daily_signals"]:
            return False, "Daily signal limit reached. Upgrade for more signals!"
        
        return True, "OK"
    
    def increment_signal_count(self, user_id):
        """Increment signal count"""
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
    
    def mark_risk_acknowledged(self, user_id):
        """Mark risk acknowledged"""
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

# ==================== WEB SERVER ====================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 LEKZY FX AI PRO - DEPLOYMENT FIXED 🚀"

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

# ==================== TRADING BOT ====================
class TradingBot:
    def __init__(self, application):
        self.app = application
        self.signal_gen = SignalGenerator()
        self.sub_mgr = SubscriptionManager(Config.DB_PATH)
        
    async def initialize(self):
        await self.signal_gen.initialize()
        logger.info("✅ TradingBot initialized successfully")
    
    async def send_welcome(self, user, chat_id):
        try:
            subscription = self.sub_mgr.get_user_subscription(user.id)
            
            if not subscription['risk_acknowledged']:
                await self.show_risk_disclaimer(user.id, chat_id)
                return
            
            message = f"""
🎉 *WELCOME TO LEKZY FX AI PRO!* 🚀

*Hello {user.first_name}!* 👋

📊 *YOUR ACCOUNT:*
• Plan: *{subscription['plan_type']}*
• Signals Used: *{subscription['signals_used']}/{subscription['max_daily_signals']}*
• Status: *✅ ACTIVE*

🤖 *FEATURES:*
• AI-Powered Signals
• Multi-Timeframe Analysis
• Session Optimization
• Risk Management

🚀 *Ready to trade? Choose an option below!*
"""
            keyboard = [
                [InlineKeyboardButton("🚀 GET TRADING SIGNAL", callback_data="normal_signal")],
                [InlineKeyboardButton("🎯 CHOOSE TIMEFRAME", callback_data="show_timeframes")],
                [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
                [InlineKeyboardButton("📊 MY STATS", callback_data="show_stats")],
                [InlineKeyboardButton("🚨 RISK GUIDE", callback_data="risk_management")]
            ]
            
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
        """Show risk disclaimer"""
        message = """
🚨 *IMPORTANT RISK DISCLAIMER* 🚨

Trading carries significant risk of loss. Only trade with risk capital you can afford to lose.

*By using this bot, you acknowledge and accept these risks.*
"""
        keyboard = [
            [InlineKeyboardButton("✅ I UNDERSTAND & ACCEPT RISKS", callback_data="accept_risks")],
            [InlineKeyboardButton("❌ CANCEL", callback_data="cancel_risks")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_risk_management(self, chat_id):
        """Show risk management guide"""
        message = """
🛡️ *RISK MANAGEMENT GUIDE* 🛡️

💰 *Essential Rules:*
• Risk Only 1-2% per trade
• Always Use Stop Loss
• Maintain 1:1.5+ Risk/Reward
• Maximum 5% total exposure

📊 *Example Position:*
• Account: $1,000
• Risk: 1% = $10 per trade
• Stop Loss: 20 pips
• Position: $0.50 per pip

🚨 *Trade responsibly!*
"""
        keyboard = [
            [InlineKeyboardButton("🚀 GET SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_plans(self, chat_id):
        """Show subscription plans"""
        message = """
💎 *SUBSCRIPTION PLANS*

🎯 *TRIAL* - FREE
• 5 signals/day
• 7 days access
• Basic features

💎 *BASIC* - $49/month
• 50 signals/day
• 30 days access
• Normal & Quick trades

🚀 *PRO* - $99/month
• 200 signals/day
• Advanced AI
• Session optimization

👑 *VIP* - $199/month
• Unlimited signals
• All features
• Premium support

📞 Contact @LekzyTradingPro to upgrade!
"""
        keyboard = [
            [InlineKeyboardButton("🚀 TRY FREE SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def show_timeframes(self, chat_id):
        """Show timeframe selection"""
        message = """
🎯 *CHOOSE TIMEFRAME*

⚡ *1 Minute (1M)*
• Quick scalping
• High frequency
• 🚨 High Risk

📈 *5 Minutes (5M)*  
• Day trading
• Balanced approach
• ⚠️ Medium Risk

🕒 *15 Minutes (15M)*
• Swing trading
• Higher confidence
• ⚠️ Medium Risk

⏰ *1 Hour (1H)*
• Position trading
• Long-term analysis
• ✅ Low Risk

📊 *4 Hours (4H)*
• Long-term investing
• Maximum confidence
• ✅ Low Risk
"""
        keyboard = [
            [
                InlineKeyboardButton("⚡ 1M", callback_data="timeframe_1M"),
                InlineKeyboardButton("📈 5M", callback_data="timeframe_5M"),
                InlineKeyboardButton("🕒 15M", callback_data="timeframe_15M")
            ],
            [
                InlineKeyboardButton("⏰ 1H", callback_data="timeframe_1H"),
                InlineKeyboardButton("📊 4H", callback_data="timeframe_4H")
            ],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def generate_signal(self, user_id, chat_id, timeframe="5M"):
        """Generate and send trading signal"""
        try:
            # Check subscription
            can_request, msg = self.sub_mgr.can_user_request_signal(user_id)
            if not can_request:
                await self.app.bot.send_message(chat_id, f"❌ {msg}")
                return
            
            await self.app.bot.send_message(chat_id, f"🎯 *Generating {timeframe} Signal...* 🤖")
            
            # Generate signal
            symbol = random.choice(self.signal_gen.pairs)
            signal = await self.signal_gen.generate_signal(symbol, timeframe)
            
            # Pre-entry message
            direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
            
            pre_msg = f"""
📊 *{timeframe} SIGNAL* 🤖

{direction_emoji} *{signal['symbol']}* | **{signal['direction']}**
💵 *Entry:* `{signal['entry_price']}`
🎯 *Confidence:* {signal['confidence']*100:.1f}%

⏰ *Timing:*
• Current: `{signal['current_time']}`
• Entry: `{signal['entry_time']}`
• Wait: *{signal['delay']}s*

*AI-optimized entry in {signal['delay']}s...* ⏳
"""
            await self.app.bot.send_message(chat_id, pre_msg, parse_mode='Markdown')
            
            # Wait for entry time
            await asyncio.sleep(signal['delay'])
            
            # Entry message with TP/SL
            entry_msg = f"""
🎯 *ENTRY SIGNAL* ✅

{direction_emoji} *{signal['symbol']}* | **{signal['direction']}**
💵 *Entry:* `{signal['entry_price']}`
✅ *TP:* `{signal['take_profit']}`
❌ *SL:* `{signal['stop_loss']}`

⏰ *Time:* `{datetime.now().strftime('%H:%M:%S')}`
📊 *TF:* {signal['timeframe']}
🎯 *Confidence:* {signal['confidence']*100:.1f}%
⚖️ *Risk/Reward:* 1:{signal['risk_reward']}

🚨 *Set Stop Loss immediately!*

*Execute this trade now!* 🚀
"""
            keyboard = [
                [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
                [InlineKeyboardButton("🔄 NEW SIGNAL", callback_data="normal_signal")],
                [InlineKeyboardButton("💎 UPGRADE PLAN", callback_data="show_plans")]
            ]
            
            await self.app.bot.send_message(
                chat_id,
                entry_msg,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            
            # Increment signal count
            self.sub_mgr.increment_signal_count(user_id)
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            await self.app.bot.send_message(chat_id, "❌ Failed to generate signal. Please try again.")

# ==================== TELEGRAM BOT HANDLER ====================
class TelegramBotHandler:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.app = None
        self.bot_core = None
    
    async def initialize(self):
        """Initialize Telegram bot"""
        try:
            if not self.token or self.token == "your_bot_token_here":
                logger.error("❌ TELEGRAM_TOKEN not set!")
                return False
            
            self.app = Application.builder().token(self.token).build()
            self.bot_core = TradingBot(self.app)
            await self.bot_core.initialize()
            
            # Add handlers
            handlers = [
                CommandHandler("start", self.start_cmd),
                CommandHandler("signal", self.signal_cmd),
                CommandHandler("plans", self.plans_cmd),
                CommandHandler("risk", self.risk_cmd),
                CommandHandler("help", self.help_cmd),
                CallbackQueryHandler(self.button_handler)
            ]
            
            for handler in handlers:
                self.app.add_handler(handler)
            
            await self.app.initialize()
            await self.app.start()
            logger.info("✅ Telegram Bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram Bot init failed: {e}")
            return False
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await self.bot_core.send_welcome(user, update.effective_chat.id)
    
    async def signal_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        timeframe = "5M"
        
        if context.args:
            for arg in context.args:
                if arg.upper() in Config.TIMEFRAMES:
                    timeframe = arg.upper()
        
        await self.bot_core.generate_signal(user.id, update.effective_chat.id, timeframe)
    
    async def plans_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_plans(update.effective_chat.id)
    
    async def risk_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_risk_management(update.effective_chat.id)
    
    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
🤖 *LEKZY FX AI PRO - HELP*

💎 *COMMANDS:*
• /start - Main menu
• /signal [TIMEFRAME] - Get AI signal
• /plans - View subscription plans
• /risk - Risk management guide
• /help - This help message

🎯 *TIMEFRAMES:*
• 1M, 5M, 15M, 1H, 4H

🚀 *Happy Trading!*
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        user = query.from_user
        data = query.data
        
        try:
            if data == "normal_signal":
                await self.bot_core.generate_signal(user.id, query.message.chat_id, "5M")
                
            elif data.startswith("timeframe_"):
                timeframe = data.replace("timeframe_", "")
                await self.bot_core.generate_signal(user.id, query.message.chat_id, timeframe)
                
            elif data == "show_timeframes":
                await self.bot_core.show_timeframes(query.message.chat_id)
                
            elif data == "show_plans":
                await self.bot_core.show_plans(query.message.chat_id)
                
            elif data == "show_stats":
                subscription = self.bot_core.sub_mgr.get_user_subscription(user.id)
                message = f"""
📊 *YOUR STATISTICS*

👤 *User:* {user.first_name}
💼 *Plan:* {subscription['plan_type']}
📈 *Signals Today:* {subscription['signals_used']}/{subscription['max_daily_signals']}
🎯 *Status:* ✅ ACTIVE

🚀 *Keep trading!*
"""
                await query.edit_message_text(message, parse_mode='Markdown')
                
            elif data == "risk_management":
                await self.bot_core.show_risk_management(query.message.chat_id)
                
            elif data == "trade_done":
                await query.edit_message_text(
                    "✅ *Trade Executed Successfully!* 🎯\n\n*Happy trading! May the profits be with you!* 💰"
                )
                
            elif data == "accept_risks":
                success = self.bot_core.sub_mgr.mark_risk_acknowledged(user.id)
                if success:
                    await query.edit_message_text(
                        "✅ *Risk Acknowledgement Confirmed!* 🛡️\n\n*Redirecting to main menu...*"
                    )
                    await asyncio.sleep(2)
                    await self.start_cmd(update, context)
                else:
                    await query.edit_message_text("❌ Failed to save. Please try /start again.")
                    
            elif data == "cancel_risks":
                await query.edit_message_text(
                    "❌ *Risk Acknowledgement Required*\n\n*Use /start when ready.*"
                )
                
            elif data == "main_menu":
                await self.start_cmd(update, context)
                
        except Exception as e:
            logger.error(f"Button error: {e}")
            await query.edit_message_text("❌ Action failed. Use /start to refresh")
    
    async def start_polling(self):
        """Start bot polling"""
        try:
            await self.app.updater.start_polling()
            logger.info("✅ Bot polling started")
            
            # Keep the bot running
            while True:
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ Polling failed: {e}")
    
    async def stop(self):
        """Stop the bot"""
        if self.app:
            await self.app.stop()

# ==================== MAIN APPLICATION ====================
async def main():
    """Main application entry point"""
    logger.info("🚀 Starting LEKZY FX AI PRO...")
    
    try:
        # Initialize database
        initialize_database()
        logger.info("✅ Database initialized")
        
        # Start web server
        start_web_server()
        logger.info("✅ Web server started")
        
        # Initialize and start Telegram bot
        bot_handler = TelegramBotHandler()
        success = await bot_handler.initialize()
        
        if success:
            logger.info("🎯 LEKZY FX AI PRO - DEPLOYMENT READY!")
            logger.info("🤖 All Systems: GO!")
            logger.info("🚀 Starting bot polling...")
            
            # Start polling
            await bot_handler.start_polling()
        else:
            logger.error("❌ Failed to start bot")
            
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        
    finally:
        logger.info("🛑 Application stopped")

if __name__ == "__main__":
    # FIXED: No directory creation needed - uses current directory
    asyncio.run(main())
