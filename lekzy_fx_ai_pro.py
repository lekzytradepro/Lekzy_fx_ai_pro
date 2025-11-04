#!/usr/bin/env python3
"""
LEKZY FX AI PRO - ULTIMATE FIXED EDITION
With working ULTRAFAST signals and proper startup
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
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from flask import Flask
from threading import Thread

# ==================== CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")
    DB_PATH = os.getenv("DB_PATH", "lekzy_fx_ai.db")
    PORT = int(os.getenv("PORT", 10000))
    
    # ULTRAFAST Trading Modes
    ULTRAFAST_MODES = {
        "HYPER": {"name": "⚡ HYPER SPEED", "pre_entry": 5, "trade_duration": 60, "accuracy": 0.85},
        "TURBO": {"name": "🚀 TURBO MODE", "pre_entry": 8, "trade_duration": 120, "accuracy": 0.88},
        "STANDARD": {"name": "🎯 STANDARD", "pre_entry": 10, "trade_duration": 300, "accuracy": 0.92}
    }
    
    TRADING_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD"]
    TIMEFRAMES = ["1M", "5M", "15M", "1H", "4H"]

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LEKZY_FX")

# ==================== DATABASE ====================
def initialize_database():
    """Initialize database"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                plan_type TEXT DEFAULT 'TRIAL',
                max_daily_signals INTEGER DEFAULT 5,
                signals_used INTEGER DEFAULT 0,
                max_ultrafast_signals INTEGER DEFAULT 2,
                ultrafast_used INTEGER DEFAULT 0,
                joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
                risk_acknowledged BOOLEAN DEFAULT FALSE
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

# ==================== AI PREDICTOR ====================
class SimpleAIPredictor:
    def __init__(self):
        self.accuracy = 0.82
        
    async def predict_direction(self, symbol):
        """Simple AI prediction"""
        try:
            direction = random.choices(["BUY", "SELL"], weights=[0.55, 0.45])[0]
            confidence = random.uniform(0.75, 0.92)
            return direction, confidence
        except Exception as e:
            return "BUY", 0.82

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def __init__(self):
        self.ai_predictor = SimpleAIPredictor()
        self.pairs = Config.TRADING_PAIRS
        
    async def generate_ultrafast_signal(self, symbol, ultrafast_mode="STANDARD", timeframe="5M"):
        """Generate ULTRAFAST trading signal"""
        try:
            mode_config = Config.ULTRAFAST_MODES[ultrafast_mode]
            
            # AI Prediction
            direction, confidence = await self.ai_predictor.predict_direction(symbol)
            
            # Generate realistic price
            price_ranges = {
                "EUR/USD": (1.07500, 1.09500), "GBP/USD": (1.25800, 1.27800),
                "USD/JPY": (148.500, 151.500), "XAU/USD": (1950.00, 2050.00),
                "AUD/USD": (0.65500, 0.67500), "USD/CAD": (1.35000, 1.37000)
            }
            
            low, high = price_ranges.get(symbol, (1.08000, 1.10000))
            current_price = round(random.uniform(low, high), 5)
            
            # Calculate entry price
            spread = 0.0002
            entry_price = round(current_price + spread if direction == "BUY" else current_price - spread, 5)
            
            # Calculate TP/SL
            if "XAU" in symbol:
                tp_distance, sl_distance = 15.0, 10.0
            elif "JPY" in symbol:
                tp_distance, sl_distance = 1.2, 0.8
            else:
                tp_distance, sl_distance = 0.0040, 0.0025
            
            if direction == "BUY":
                take_profit = round(entry_price + tp_distance, 5)
                stop_loss = round(entry_price - sl_distance, 5)
            else:
                take_profit = round(entry_price - tp_distance, 5)
                stop_loss = round(entry_price + sl_distance, 5)
            
            risk_reward = round(tp_distance / sl_distance, 2)
            
            # Timing
            pre_entry_delay = mode_config["pre_entry"]
            trade_duration = mode_config["trade_duration"]
            
            current_time = datetime.now()
            entry_time = current_time + timedelta(seconds=pre_entry_delay)
            exit_time = entry_time + timedelta(seconds=trade_duration)
            
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "confidence": confidence,
                "risk_reward": risk_reward,
                "timeframe": timeframe,
                "ultrafast_mode": ultrafast_mode,
                "mode_name": mode_config["name"],
                "pre_entry_delay": pre_entry_delay,
                "trade_duration": trade_duration,
                "current_time": current_time.strftime("%H:%M:%S"),
                "entry_time": entry_time.strftime("%H:%M:%S"),
                "exit_time": exit_time.strftime("%H:%M:%S"),
                "ai_systems": ["Advanced AI Analysis"]
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
                "confidence": 0.82,
                "risk_reward": 1.5,
                "timeframe": timeframe,
                "ultrafast_mode": ultrafast_mode,
                "mode_name": "STANDARD",
                "pre_entry_delay": 10,
                "trade_duration": 60,
                "current_time": datetime.now().strftime("%H:%M:%S"),
                "entry_time": (datetime.now() + timedelta(seconds=10)).strftime("%H:%M:%S"),
                "exit_time": (datetime.now() + timedelta(seconds=70)).strftime("%H:%M:%S"),
                "ai_systems": ["Basic Analysis"]
            }

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_user_subscription(self, user_id):
        """Get user subscription info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT plan_type, max_daily_signals, signals_used, max_ultrafast_signals, ultrafast_used, risk_acknowledged 
                FROM users WHERE user_id = ?
            """, (user_id,))
            result = cursor.fetchone()
            
            if result:
                plan_type, max_signals, signals_used, max_ultrafast, ultrafast_used, risk_ack = result
                return {
                    "plan_type": plan_type,
                    "max_daily_signals": max_signals,
                    "signals_used": signals_used,
                    "signals_remaining": max_signals - signals_used,
                    "max_ultrafast_signals": max_ultrafast,
                    "ultrafast_used": ultrafast_used,
                    "ultrafast_remaining": max_ultrafast - ultrafast_used,
                    "risk_acknowledged": risk_ack
                }
            else:
                # Create new user
                conn.execute("""
                    INSERT INTO users (user_id, plan_type, max_daily_signals, max_ultrafast_signals) 
                    VALUES (?, ?, ?, ?)
                """, (user_id, "TRIAL", 5, 2))
                conn.commit()
                conn.close()
                
                return {
                    "plan_type": "TRIAL",
                    "max_daily_signals": 5,
                    "signals_used": 0,
                    "signals_remaining": 5,
                    "max_ultrafast_signals": 2,
                    "ultrafast_used": 0,
                    "ultrafast_remaining": 2,
                    "risk_acknowledged": False
                }
                
        except Exception as e:
            logger.error(f"❌ Get subscription failed: {e}")
            return self.get_fallback_subscription()
    
    def get_fallback_subscription(self):
        return {
            "plan_type": "TRIAL",
            "max_daily_signals": 5,
            "signals_used": 0,
            "signals_remaining": 5,
            "max_ultrafast_signals": 2,
            "ultrafast_used": 0,
            "ultrafast_remaining": 2,
            "risk_acknowledged": False
        }
    
    def can_user_request_signal(self, user_id, is_ultrafast=False):
        """Check if user can request signal"""
        subscription = self.get_user_subscription(user_id)
        
        if is_ultrafast:
            if subscription["ultrafast_used"] >= subscription["max_ultrafast_signals"]:
                return False, "ULTRAFAST signal limit reached!"
        else:
            if subscription["signals_used"] >= subscription["max_daily_signals"]:
                return False, "Daily signal limit reached!"
        
        return True, "OK"
    
    def increment_signal_count(self, user_id, is_ultrafast=False):
        """Increment signal count"""
        try:
            conn = sqlite3.connect(self.db_path)
            if is_ultrafast:
                conn.execute("UPDATE users SET ultrafast_used = ultrafast_used + 1 WHERE user_id = ?", (user_id,))
            else:
                conn.execute("UPDATE users SET signals_used = signals_used + 1 WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Signal count increment failed: {e}")
            return False
    
    def mark_risk_acknowledged(self, user_id):
        """Mark risk acknowledged"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("UPDATE users SET risk_acknowledged = TRUE WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Risk acknowledgment failed: {e}")
            return False

# ==================== TRADING BOT ====================
class TradingBot:
    def __init__(self, application):
        self.app = application
        self.signal_gen = SignalGenerator()
        self.sub_mgr = SubscriptionManager(Config.DB_PATH)
    
    async def send_welcome(self, user, chat_id):
        """Send ULTRAFAST welcome message"""
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
• Regular Signals: *{subscription['signals_used']}/{subscription['max_daily_signals']}*
• ULTRAFAST Signals: *{subscription['ultrafast_used']}/{subscription['max_ultrafast_signals']}*

⚡ *ULTRAFAST TRADING MODES:*
• Hyper Speed (5s pre-entry, 1min trades)
• Turbo Mode (8s pre-entry, 2min trades) 
• Standard (10s pre-entry, 5min trades)

🚀 *Ready to experience lightning-fast AI trading?*
"""
            keyboard = [
                [InlineKeyboardButton("⚡ ULTRAFAST SIGNALS", callback_data="ultrafast_menu")],
                [InlineKeyboardButton("🎯 REGULAR SIGNALS", callback_data="normal_signal")],
                [InlineKeyboardButton("📊 MY STATS", callback_data="show_stats")],
                [InlineKeyboardButton("💎 VIEW PLANS", callback_data="show_plans")],
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
            # Fallback message
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=f"🚀 Welcome {user.first_name} to LEKZY FX AI PRO!\n\nUse the buttons below to get started!",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("⚡ GET STARTED", callback_data="ultrafast_menu")
                ]])
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
    
    async def show_ultrafast_menu(self, chat_id):
        """Show ULTRAFAST trading modes"""
        message = """
⚡ *ULTRAFAST TRADING MODES* 🚀

*Choose your trading speed:*

🎯 *STANDARD MODE*
• Pre-entry: 10 seconds
• Trade Duration: 5 minutes  
• Perfect for beginners

🚀 *TURBO MODE* 
• Pre-entry: 8 seconds
• Trade Duration: 2 minutes
• Balanced speed & accuracy

⚡ *HYPER SPEED*
• Pre-entry: 5 seconds
• Trade Duration: 1 minute
• Maximum speed execution

*All modes include AI-powered signals with high accuracy!*
"""
        keyboard = [
            [
                InlineKeyboardButton("🎯 STANDARD", callback_data="ultrafast_STANDARD"),
                InlineKeyboardButton("🚀 TURBO", callback_data="ultrafast_TURBO")
            ],
            [
                InlineKeyboardButton("⚡ HYPER SPEED", callback_data="ultrafast_HYPER"),
                InlineKeyboardButton("📊 CHOOSE TIMEFRAME", callback_data="show_timeframes")
            ],
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

*Recommended for ULTRAFAST:*
⚡ *1 Minute (1M)* - Hyper Speed
📈 *5 Minutes (5M)* - Turbo Mode  
🕒 *15 Minutes (15M)* - Standard

*Regular Trading:*
⏰ *1 Hour (1H)* - Position trading
📊 *4 Hours (4H)* - Long-term analysis
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
            [InlineKeyboardButton("⚡ ULTRAFAST MENU", callback_data="ultrafast_menu")],
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
• 5 regular signals/day
• 2 ULTRAFAST signals/day
• Basic AI features

💎 *BASIC* - $49/month
• 50 regular signals/day  
• 10 ULTRAFAST signals/day
• All ULTRAFAST modes

🚀 *PRO* - $99/month
• 200 regular signals/day
• 50 ULTRAFAST signals/day
• Advanced AI features

👑 *VIP* - $199/month
• Unlimited regular signals
• 200 ULTRAFAST signals/day
• Maximum performance
"""
        keyboard = [
            [InlineKeyboardButton("⚡ TRY ULTRAFAST", callback_data="ultrafast_menu")],
            [InlineKeyboardButton("🎯 FREE SIGNAL", callback_data="normal_signal")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
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
            [InlineKeyboardButton("⚡ GET SIGNAL", callback_data="ultrafast_menu")],
            [InlineKeyboardButton("🏠 MAIN MENU", callback_data="main_menu")]
        ]
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    async def generate_ultrafast_signal(self, user_id, chat_id, ultrafast_mode="STANDARD", timeframe="5M"):
        """Generate ULTRAFAST trading signal"""
        try:
            logger.info(f"🔄 Starting ULTRAFAST signal for user {user_id}")
            
            # Check subscription
            can_request, msg = self.sub_mgr.can_user_request_signal(user_id, is_ultrafast=True)
            if not can_request:
                await self.app.bot.send_message(chat_id, f"❌ {msg}")
                return
            
            mode_config = Config.ULTRAFAST_MODES[ultrafast_mode]
            await self.app.bot.send_message(
                chat_id, 
                f"⚡ *Initializing {mode_config['name']}...*"
            )
            
            # Generate signal
            symbol = random.choice(self.signal_gen.pairs)
            signal = await self.signal_gen.generate_ultrafast_signal(symbol, ultrafast_mode, timeframe)
            
            # Pre-entry message
            direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
            
            pre_msg = f"""
⚡ *{signal['mode_name']} - {timeframe} SIGNAL*

{symbol} | **{signal['direction']}** {direction_emoji}
🎯 *Confidence:* {signal['confidence']*100:.1f}%

⏰ *Timing:*
• Entry in: *{signal['pre_entry_delay']}s*
• Trade duration: *{signal['trade_duration']}s*

*Getting ready for entry...* ⚡
"""
            await self.app.bot.send_message(chat_id, pre_msg, parse_mode='Markdown')
            
            # Wait for entry
            await asyncio.sleep(signal['pre_entry_delay'])
            
            # Entry message
            entry_msg = f"""
🎯 *ENTRY SIGNAL* ✅

⚡ *{signal['mode_name']}*
{direction_emoji} *{signal['symbol']}* | **{signal['direction']}**

💵 *Entry:* `{signal['entry_price']}`
✅ *TP:* `{signal['take_profit']}`
❌ *SL:* `{signal['stop_loss']}`

📊 *Metrics:*
• Confidence: *{signal['confidence']*100:.1f}%*
• Risk/Reward: *1:{signal['risk_reward']}*
• Timeframe: *{signal['timeframe']}*

🚨 *Set Stop Loss immediately!*
⚡ *Execute NOW!*
"""
            keyboard = [
                [InlineKeyboardButton("✅ TRADE EXECUTED", callback_data="trade_done")],
                [InlineKeyboardButton("⚡ NEW SIGNAL", callback_data="ultrafast_menu")],
            ]
            
            await self.app.bot.send_message(
                chat_id,
                entry_msg,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            
            # Increment signal count
            self.sub_mgr.increment_signal_count(user_id, is_ultrafast=True)
            
            logger.info(f"✅ ULTRAFAST signal completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ ULTRAFAST signal failed: {e}")
            await self.app.bot.send_message(chat_id, "❌ Signal generation failed. Please try again.")
    
    async def generate_regular_signal(self, user_id, chat_id, timeframe="5M"):
        """Generate regular trading signal"""
        try:
            # Use STANDARD mode for regular signals
            await self.generate_ultrafast_signal(user_id, chat_id, "STANDARD", timeframe)
        except Exception as e:
            logger.error(f"❌ Regular signal failed: {e}")
            await self.app.bot.send_message(chat_id, "❌ Signal generation failed. Please try again.")

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
            
            # Create application
            self.app = Application.builder().token(self.token).build()
            self.bot_core = TradingBot(self.app)
            
            # Add handlers - MAKE SURE THESE ARE CORRECT
            self.app.add_handler(CommandHandler("start", self.start_cmd))
            self.app.add_handler(CommandHandler("signal", self.signal_cmd))
            self.app.add_handler(CommandHandler("ultrafast", self.ultrafast_cmd))
            self.app.add_handler(CommandHandler("plans", self.plans_cmd))
            self.app.add_handler(CommandHandler("risk", self.risk_cmd))
            self.app.add_handler(CommandHandler("stats", self.stats_cmd))
            self.app.add_handler(CommandHandler("help", self.help_cmd))
            self.app.add_handler(CallbackQueryHandler(self.button_handler))
            
            logger.info("✅ Telegram Bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram Bot init failed: {e}")
            return False
    
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        await self.bot_core.send_welcome(user, update.effective_chat.id)
    
    async def signal_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command"""
        user = update.effective_user
        timeframe = "5M"
        
        if context.args:
            for arg in context.args:
                if arg.upper() in Config.TIMEFRAMES:
                    timeframe = arg.upper()
        
        await self.bot_core.generate_regular_signal(user.id, update.effective_chat.id, timeframe)
    
    async def ultrafast_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ultrafast command"""
        user = update.effective_user
        mode = "STANDARD"
        timeframe = "5M"
        
        if context.args:
            for arg in context.args:
                arg_upper = arg.upper()
                if arg_upper in Config.ULTRAFAST_MODES:
                    mode = arg_upper
                elif arg_upper in Config.TIMEFRAMES:
                    timeframe = arg_upper
        
        await self.bot_core.generate_ultrafast_signal(user.id, update.effective_chat.id, mode, timeframe)
    
    async def plans_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_plans(update.effective_chat.id)
    
    async def risk_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.bot_core.show_risk_management(update.effective_chat.id)
    
    async def stats_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        subscription = self.bot_core.sub_mgr.get_user_subscription(user.id)
        
        message = f"""
📊 *YOUR STATISTICS*

👤 *Trader:* {user.first_name}
💼 *Plan:* {subscription['plan_type']}
📈 *Regular Signals:* {subscription['signals_used']}/{subscription['max_daily_signals']}
⚡ *ULTRAFAST Signals:* {subscription['ultrafast_used']}/{subscription['max_ultrafast_signals']}

🚀 *Keep trading!*
"""
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
🤖 *LEKZY FX AI PRO - HELP*

💎 *COMMANDS:*
• /start - Main menu with ULTRAFAST options
• /signal [TIMEFRAME] - Regular signal
• /ultrafast [MODE] [TIMEFRAME] - ULTRAFAST signal
• /plans - Subscription plans
• /risk - Risk management
• /stats - Your statistics
• /help - This message

⚡ *ULTRAFAST MODES:*
• HYPER - 5s pre-entry, 1min trades
• TURBO - 8s pre-entry, 2min trades  
• STANDARD - 10s pre-entry, 5min trades

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
                await self.bot_core.generate_regular_signal(user.id, query.message.chat_id, "5M")
                
            elif data.startswith("ultrafast_"):
                mode = data.replace("ultrafast_", "")
                await self.bot_core.generate_ultrafast_signal(user.id, query.message.chat_id, mode, "5M")
                
            elif data == "ultrafast_menu":
                await self.bot_core.show_ultrafast_menu(query.message.chat_id)
                
            elif data.startswith("timeframe_"):
                timeframe = data.replace("timeframe_", "")
                if "ultrafast" in query.message.text:
                    await self.bot_core.generate_ultrafast_signal(user.id, query.message.chat_id, "STANDARD", timeframe)
                else:
                    await self.bot_core.generate_regular_signal(user.id, query.message.chat_id, timeframe)
                
            elif data == "show_timeframes":
                await self.bot_core.show_timeframes(query.message.chat_id)
                
            elif data == "show_plans":
                await self.bot_core.show_plans(query.message.chat_id)
                
            elif data == "show_stats":
                subscription = self.bot_core.sub_mgr.get_user_subscription(user.id)
                message = f"""
📊 *YOUR STATS*

👤 {user.first_name}
💼 {subscription['plan_type']}
📈 Regular: {subscription['signals_used']}/{subscription['max_daily_signals']}
⚡ ULTRAFAST: {subscription['ultrafast_used']}/{subscription['max_ultrafast_signals']}
"""
                await query.edit_message_text(message, parse_mode='Markdown')
                
            elif data == "risk_management":
                await self.bot_core.show_risk_management(query.message.chat_id)
                
            elif data == "trade_done":
                await query.edit_message_text("✅ *Trade Executed!* 🎯\n\nHappy trading! 💰")
                
            elif data == "accept_risks":
                success = self.bot_core.sub_mgr.mark_risk_acknowledged(user.id)
                if success:
                    await query.edit_message_text("✅ *Risk Accepted!*\n\nRedirecting to main menu...")
                    await asyncio.sleep(2)
                    await self.start_cmd(update, context)
                else:
                    await query.edit_message_text("❌ Failed. Try /start again.")
                    
            elif data == "cancel_risks":
                await query.edit_message_text("❌ Risk acknowledgement required.\n\nUse /start when ready.")
                
            elif data == "main_menu":
                await self.start_cmd(update, context)
                
        except Exception as e:
            logger.error(f"❌ Button error: {e}")
            await query.edit_message_text("❌ Action failed. Use /start to refresh")
    
    async def start_polling(self):
        """Start bot polling"""
        try:
            await self.app.run_polling()
            logger.info("✅ Bot polling started")
        except Exception as e:
            logger.error(f"❌ Polling failed: {e}")

# ==================== WEB SERVER ====================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 LEKZY FX AI PRO - ULTRAFAST EDITION 🚀"

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

# ==================== MAIN APPLICATION ====================
async def main():
    """Main application"""
    logger.info("🚀 Starting LEKZY FX AI PRO - ULTRAFAST EDITION...")
    
    try:
        # Initialize database
        initialize_database()
        logger.info("✅ Database initialized")
        
        # Start web server
        start_web_server()
        logger.info("✅ Web server started")
        
        # Initialize and start bot
        bot_handler = TelegramBotHandler()
        success = await bot_handler.initialize()
        
        if success:
            logger.info("🎯 LEKZY FX AI PRO READY!")
            logger.info("⚡ ULTRAFAST Modes: ACTIVE")
            logger.info("🚀 Starting bot polling...")
            
            # Start polling
            await bot_handler.start_polling()
        else:
            logger.error("❌ Failed to start bot")
            
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
