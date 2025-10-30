import os
import asyncio
import aiohttp
import sqlite3
import json
import time
import random
import threading
import logging
import signal
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

# Configuration and environment
from dotenv import load_dotenv

# Telegram imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ML imports with graceful fallbacks
try:
    import numpy as np
    import joblib
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available - ML features disabled")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available - using fallback classifiers")

# -------------------- Configuration --------------------
load_dotenv()

class Config:
    """Centralized configuration management"""
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
    DB_PATH = os.getenv("DB_PATH", "trade_data.db")
    MODEL_DIR = os.getenv("MODEL_DIR", "models")
    SCALER_DIR = os.getenv("SCALER_DIR", "scalers")
    RETRAIN_CANDLES = int(os.getenv("RETRAIN_CANDLES", "200"))
    PREENTRY_DEFAULT = int(os.getenv("PREENTRY_DEFAULT", "30"))
    HTTP_PORT = int(os.getenv("PORT", "8080"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    RENDER = os.getenv("RENDER", "false").lower() == "true"
    
    # Trading parameters
    MIN_SIGNAL_COOLDOWN = 40
    MAX_SIGNAL_COOLDOWN = 180
    CANDLE_LIMIT = 500
    TWELVE_CACHE_TTL = 15
    
    # Volatility filters
    VOLATILITY_MIN_FOREX = 0.00025
    VOLATILITY_MAX_FOREX = 0.006
    VOLATILITY_MIN_CRYPTO = 12
    VOLATILITY_MAX_CRYPTO = 800
    
    # Timezone
    TZ = pytz.timezone("Etc/GMT-1")  # UTC+1

    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        if not cls.TELEGRAM_TOKEN:
            raise RuntimeError("TELEGRAM_TOKEN is required in environment variables")
        
        if not cls.ADMIN_TOKEN:
            logging.warning("ADMIN_TOKEN is not set. Admin commands will be disabled")
        
        # Ensure directories exist
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SCALER_DIR, exist_ok=True)

# Initialize configuration
Config.validate()

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lekzy_fx_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LekzyFXAI")

# -------------------- Web Server for Render --------------------
class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks"""
    
    def do_GET(self):
        if self.path in ("/", "/health", "/status"):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "LekzyFXAIPro",
                "timestamp": datetime.now(Config.TZ).isoformat(),
                "version": "1.0.0",
                "environment": "production" if Config.RENDER else "development"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        logger.debug(f"HTTP {format % args}")

class WebServer:
    """Web server to keep the app alive on Render"""
    
    def __init__(self, port: int = Config.HTTP_PORT):
        self.port = port
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def start(self):
        """Start the web server in a background thread"""
        def run_server():
            try:
                self.server = HTTPServer(('0.0.0.0', self.port), HealthHandler)
                self.is_running = True
                logger.info(f"Web server started on port {self.port}")
                self.server.serve_forever()
            except Exception as e:
                logger.error(f"Web server error: {e}")
            finally:
                self.is_running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=False)
        self.server_thread.start()
        
        # Wait a bit to ensure server starts
        time.sleep(2)
        if self.is_running:
            logger.info("Web server is ready and accepting requests")
        else:
            logger.error("Web server failed to start")
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Web server stopped")
        self.is_running = False

# -------------------- Simplified Trading Bot --------------------
class LekzyFXAIPro:
    """Main trading bot class"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._init_db()
        self.user_sessions = {}
        self.performance_stats = {
            'total_signals': 0,
            'active_users': 0,
            'start_time': datetime.now(Config.TZ)
        }
        logger.info("LekzyFXAIPro initialized")
    
    def _init_db(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE,
                    symbol TEXT,
                    side TEXT,
                    timeframe TEXT,
                    entry_price REAL,
                    confidence REAL,
                    timestamp TEXT,
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authorized_users (
                    chat_id INTEGER PRIMARY KEY,
                    username TEXT,
                    authorized_at TEXT
                )
            """)
            conn.commit()
    
    async def start_user_session(self, user_id: int) -> bool:
        """Start trading session for user"""
        if user_id in self.user_sessions:
            return False
        
        self.user_sessions[user_id] = {
            'start_time': datetime.now(Config.TZ),
            'signals_sent': 0,
            'active': True
        }
        self.performance_stats['active_users'] += 1
        logger.info(f"Started session for user {user_id}")
        return True
    
    async def stop_user_session(self, user_id: int) -> bool:
        """Stop trading session for user"""
        if user_id not in self.user_sessions:
            return False
        
        self.user_sessions[user_id]['active'] = False
        self.performance_stats['active_users'] -= 1
        logger.info(f"Stopped session for user {user_id}")
        return True

# -------------------- Telegram Bot --------------------
class TelegramBot:
    """Telegram bot handler"""
    
    def __init__(self, trading_bot: LekzyFXAIPro):
        self.bot = trading_bot
        self.application = None
    
    async def initialize(self):
        """Initialize Telegram bot"""
        if not Config.TELEGRAM_TOKEN:
            raise RuntimeError("TELEGRAM_TOKEN not set")
        
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self._setup_handlers()
        await self.application.initialize()
        await self.application.start()
        logger.info("Telegram bot initialized")
    
    def _setup_handlers(self):
        """Setup command handlers"""
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("stats", self._stats_command))
        self.application.add_handler(CommandHandler("stop", self._stop_command))
        self.application.add_handler(CallbackQueryHandler(self._button_handler))
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_text = (
            "🤖 *Lekzy FX AI Pro*\n\n"
            "AI-powered trading signals\n\n"
            "Commands:\n"
            "• /start - Show this message\n"
            "• /stats - Show statistics\n"
            "• /stop - Stop signals\n"
        )
        
        keyboard = [
            [InlineKeyboardButton("🚀 Start Signals", callback_data="start_signals")],
            [InlineKeyboardButton("📊 Stats", callback_data="live_stats")],
            [InlineKeyboardButton("🛑 Stop", callback_data="stop_signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        stats = self.bot.performance_stats
        uptime = datetime.now(Config.TZ) - stats['start_time']
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats_text = (
            f"📊 *Statistics*\n\n"
            f"• Active Users: {stats['active_users']}\n"
            f"• Total Signals: {stats['total_signals']}\n"
            f"• Uptime: {hours}h {minutes}m\n"
        )
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def _stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        user_id = update.effective_user.id
        success = await self.bot.stop_user_session(user_id)
        if success:
            await update.message.reply_text("🛑 Signals stopped.")
        else:
            await update.message.reply_text("ℹ️ No active session found.")
    
    async def _button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses"""
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        if query.data == "start_signals":
            success = await self.bot.start_user_session(user_id)
            if success:
                await query.edit_message_text("✅ Signals started!")
            else:
                await query.edit_message_text("❌ Already running!")
        
        elif query.data == "stop_signals":
            success = await self.bot.stop_user_session(user_id)
            if success:
                await query.edit_message_text("🛑 Signals stopped.")
            else:
                await query.edit_message_text("❌ No active session.")
        
        elif query.data == "live_stats":
            stats = self.bot.performance_stats
            await query.edit_message_text(f"Active users: {stats['active_users']}\nTotal signals: {stats['total_signals']}")
    
    async def start_polling(self):
        """Start polling for updates"""
        if not self.application:
            raise RuntimeError("Telegram bot not initialized")
        
        logger.info("Starting Telegram bot polling...")
        await self.application.updater.start_polling(
            poll_interval=1.0,
            timeout=10,
            drop_pending_updates=True
        )
    
    async def stop_polling(self):
        """Stop polling"""
        if self.application and self.application.updater:
            await self.application.updater.stop()
            logger.info("Telegram polling stopped")
    
    async def shutdown(self):
        """Shutdown Telegram bot"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot shutdown complete")

# -------------------- Main Application --------------------
class ApplicationManager:
    """Main application manager"""
    
    def __init__(self):
        self.trading_bot = None
        self.telegram_bot = None
        self.web_server = None
        self.is_running = False
    
    async def setup(self):
        """Setup application components"""
        logger.info("Setting up Lekzy FX AI Pro...")
        
        # Initialize trading bot
        self.trading_bot = LekzyFXAIPro()
        
        # Initialize Telegram bot
        self.telegram_bot = TelegramBot(self.trading_bot)
        await self.telegram_bot.initialize()
        
        # Start web server
        self.web_server = WebServer()
        self.web_server.start()
        
        self.is_running = True
        logger.info("Application setup completed successfully")
    
    async def run(self):
        """Run the main application"""
        if not self.is_running:
            raise RuntimeError("Application not properly initialized")
        
        logger.info("Starting application...")
        
        try:
            # Start Telegram polling in background
            polling_task = asyncio.create_task(self.telegram_bot.start_polling())
            
            # Keep the main thread alive
            while self.is_running:
                await asyncio.sleep(1)
                
                # Send periodic heartbeat signal
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.debug("Application heartbeat - running normally")
            
            # Cleanup if we break out of the loop
            await polling_task
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        if not self.is_running:
            return
        
        logger.info("Initiating shutdown...")
        self.is_running = False
        
        try:
            # Stop Telegram bot
            if self.telegram_bot:
                await self.telegram_bot.shutdown()
            
            # Stop web server
            if self.web_server:
                self.web_server.stop()
            
            logger.info("Shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# -------------------- Signal Handlers --------------------
def setup_signal_handlers(app_manager):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(app_manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# -------------------- Main Execution --------------------
async def main():
    """Main application entry point"""
    app_manager = ApplicationManager()
    
    try:
        # Setup signal handlers
        setup_signal_handlers(app_manager)
        
        # Setup application
        await app_manager.setup()
        
        # Run application
        await app_manager.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"Application error: {e}")
        raise
    finally:
        # Ensure cleanup
        await app_manager.shutdown()

if __name__ == "__main__":
    logger.info("Starting Lekzy FX AI Pro Application")
    logger.info(f"Render mode: {Config.RENDER}")
    logger.info(f"Web server port: {Config.HTTP_PORT}")
    
    try:
        # Run the application
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        exit(1)
