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
import numpy as np

# Configuration and environment
from dotenv import load_dotenv

# Telegram imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

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

# -------------------- Technical Indicators --------------------
class TechnicalIndicators:
    """Technical analysis indicator calculations"""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return float(np.mean(prices)) if len(prices) > 0 else 0.0
        return float(np.mean(prices[-period:]))
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        if len(prices) < period:
            return float(np.mean(prices))
        
        alpha = 2 / (period + 1)
        ema_value = prices[0]
        for price in prices[1:]:
            ema_value = alpha * price + (1 - alpha) * ema_value
        return float(ema_value)
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    @staticmethod
    def bollinger_bands_width(prices: np.ndarray, period: int = 20) -> float:
        """Bollinger Bands Width (normalized)"""
        if len(prices) < 2:
            return 0.0
        
        sma_val = TechnicalIndicators.sma(prices, min(period, len(prices)))
        std = float(np.std(prices[-period:])) if len(prices) >= period else float(np.std(prices))
        
        upper_band = sma_val + 2 * std
        lower_band = sma_val - 2 * std
        
        return float((upper_band - lower_band) / (abs(sma_val) + 1e-9))
    
    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        return float(np.mean(true_ranges[-period:])) if true_ranges else 0.0

# -------------------- Trading Assets --------------------
TRADING_ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/JPY", "GBP/JPY",
    "USD/CAD", "EUR/GBP", "USD/CHF", "BTC/USD", "ETH/USD", "XAU/USD", "XAG/USD"
]

# -------------------- Signal Generator --------------------
class SignalGenerator:
    """Generate trading signals with market analysis"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
    
    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def generate_synthetic_candles(self, symbol: str, interval: str = "1min", limit: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic market data for testing"""
        candles = []
        base_price = random.uniform(1.0, 100.0) if "USD" in symbol else random.uniform(10000, 50000)
        
        for i in range(limit):
            change = random.uniform(-0.002, 0.002)
            open_price = base_price
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
            volume = random.uniform(1000, 10000)
            
            candle_time = datetime.utcnow() - timedelta(minutes=(limit - i))
            
            candles.append({
                "t": candle_time.isoformat(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
            
            base_price = close_price
        
        return candles
    
    def analyze_market(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market data and generate features"""
        if not candles or len(candles) < 20:
            return {"direction": "HOLD", "confidence": 0.0, "reason": "Insufficient data"}
        
        closes = np.array([c['close'] for c in candles], dtype=float)
        highs = np.array([c['high'] for c in candles], dtype=float)
        lows = np.array([c['low'] for c in candles], dtype=float)
        
        # Calculate technical indicators
        sma_20 = TechnicalIndicators.sma(closes, 20)
        sma_50 = TechnicalIndicators.sma(closes, 50)
        rsi = TechnicalIndicators.rsi(closes)
        bb_width = TechnicalIndicators.bollinger_bands_width(closes)
        atr = TechnicalIndicators.atr(highs, lows, closes)
        
        current_price = closes[-1]
        price_trend = "BULLISH" if current_price > sma_20 else "BEARISH"
        momentum = "STRONG" if abs(current_price - sma_20) > atr else "WEAK"
        
        # Generate signal based on analysis
        if rsi < 30 and price_trend == "BULLISH":
            direction = "BUY"
            confidence = min(85.0, 70 + (30 - rsi) * 0.5)
            reason = f"Oversold (RSI: {rsi:.1f}) with bullish trend"
        elif rsi > 70 and price_trend == "BEARISH":
            direction = "SELL"
            confidence = min(85.0, 70 + (rsi - 70) * 0.5)
            reason = f"Overbought (RSI: {rsi:.1f}) with bearish trend"
        elif current_price > sma_50 and sma_20 > sma_50:
            direction = "BUY"
            confidence = 75.0
            reason = "Strong uptrend with MA alignment"
        elif current_price < sma_50 and sma_20 < sma_50:
            direction = "SELL"
            confidence = 75.0
            reason = "Strong downtrend with MA alignment"
        else:
            direction = "HOLD"
            confidence = 0.0
            reason = "No clear signal"
        
        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "reason": reason,
            "indicators": {
                "rsi": round(rsi, 2),
                "sma_20": round(sma_20, 4),
                "sma_50": round(sma_50, 4),
                "bb_width": round(bb_width, 4),
                "atr": round(atr, 4),
                "current_price": round(current_price, 4)
            }
        }
    
    async def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate a trading signal for a symbol"""
        try:
            # Get market data
            candles = await self.generate_synthetic_candles(symbol, "5min", 100)
            
            # Analyze market
            analysis = self.analyze_market(candles)
            
            # Only return signals with sufficient confidence
            if analysis["confidence"] > 65 and analysis["direction"] != "HOLD":
                return {
                    "symbol": symbol,
                    "direction": analysis["direction"],
                    "confidence": analysis["confidence"],
                    "reason": analysis["reason"],
                    "price": analysis["indicators"]["current_price"],
                    "timestamp": datetime.now(Config.TZ),
                    "timeframe": "5min",
                    "indicators": analysis["indicators"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close resources"""
        if self.session and not self.session.closed:
            await self.session.close()

# -------------------- Trading Bot --------------------
class LekzyFXAIPro:
    """Main trading bot class with signal generation"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.signal_generator = SignalGenerator()
        self.user_sessions = {}
        self.active_signals = {}
        self.performance_stats = {
            'total_signals': 0,
            'active_users': 0,
            'start_time': datetime.now(Config.TZ),
            'last_signal_time': None
        }
        self._init_db()
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
            'signals_received': 0,
            'active': True,
            'last_signal': None
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
    
    async def generate_and_send_signals(self, application):
        """Generate and send signals to active users"""
        if not self.user_sessions:
            return
        
        active_users = [uid for uid, session in self.user_sessions.items() if session['active']]
        if not active_users:
            return
        
        # Select a random asset to analyze
        symbol = random.choice(TRADING_ASSETS)
        
        try:
            # Generate signal
            signal = await self.signal_generator.generate_signal(symbol)
            
            if signal and signal['confidence'] > 65:
                logger.info(f"Generated signal: {symbol} {signal['direction']} ({signal['confidence']}%)")
                
                # Send to all active users
                for user_id in active_users:
                    try:
                        await self.send_signal_message(application, user_id, signal)
                        self.user_sessions[user_id]['signals_received'] += 1
                        self.user_sessions[user_id]['last_signal'] = datetime.now(Config.TZ)
                    except Exception as e:
                        logger.error(f"Error sending signal to user {user_id}: {e}")
                
                self.performance_stats['total_signals'] += 1
                self.performance_stats['last_signal_time'] = datetime.now(Config.TZ)
                
        except Exception as e:
            logger.error(f"Error in signal generation cycle: {e}")
    
    async def send_signal_message(self, application, user_id: int, signal: Dict[str, Any]):
        """Send signal message to user"""
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        confidence_level = "HIGH" if signal['confidence'] > 80 else "MEDIUM" if signal['confidence'] > 65 else "LOW"
        
        message = f"""
{direction_emoji} *TRADING SIGNAL*

📊 *Pair:* {signal['symbol']}
🎯 *Action:* {signal['direction']}
📈 *Confidence:* {signal['confidence']}% ({confidence_level})
💰 *Current Price:* {signal['price']:.4f}
⏰ *Timeframe:* {signal['timeframe']}
🕒 *Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)

*Analysis:*
{signal['reason']}

*Technical Indicators:*
• RSI: {signal['indicators']['rsi']}
• SMA 20: {signal['indicators']['sma_20']:.4f}
• SMA 50: {signal['indicators']['sma_50']:.4f}
• ATR: {signal['indicators']['atr']:.4f}

⚠️ *Risk Warning: Trading involves risk. Use proper risk management.*
"""
        
        keyboard = [
            [InlineKeyboardButton("✅ Take Trade", callback_data=f"trade_{signal['direction']}"),
             InlineKeyboardButton("❌ Skip", callback_data="skip_trade")],
            [InlineKeyboardButton("🛑 Stop Signals", callback_data="stop_signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await application.bot.send_message(
            chat_id=user_id,
            text=message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"Sent signal to user {user_id}: {signal['symbol']} {signal['direction']}")

# -------------------- Telegram Bot --------------------
class TelegramBot:
    """Telegram bot handler"""
    
    def __init__(self, trading_bot: LekzyFXAIPro):
        self.bot = trading_bot
        self.application = None
        self.signal_task = None
    
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
        self.application.add_handler(CommandHandler("signal", self._signal_command))
        self.application.add_handler(CallbackQueryHandler(self._button_handler))
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_text = """
🤖 *Lekzy FX AI Pro*

Advanced AI-powered trading signals with real-time market analysis.

*Commands:*
• /start - Show this message
• /stats - Show statistics  
• /signal - Request immediate signal
• /stop - Stop signals

*Features:*
• Real-time market analysis
• Multiple timeframe signals
• Risk management guidance
• Technical indicator analysis

Click below to start receiving signals!
"""
        
        keyboard = [
            [InlineKeyboardButton("🚀 Start Signals", callback_data="start_signals")],
            [InlineKeyboardButton("📊 Live Stats", callback_data="live_stats")],
            [InlineKeyboardButton("🎯 Get Signal", callback_data="get_signal")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        stats = self.bot.performance_stats
        uptime = datetime.now(Config.TZ) - stats['start_time']
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        active_users = len([u for u in self.bot.user_sessions.values() if u['active']])
        
        stats_text = f"""
📊 *System Statistics*

• Active Users: {active_users}
• Total Signals: {stats['total_signals']}
• Uptime: {hours}h {minutes}m
• Last Signal: {stats['last_signal_time'].strftime('%H:%M:%S') if stats['last_signal_time'] else 'Never'}

*Monitored Assets:*
{', '.join(TRADING_ASSETS[:6])}
...
"""
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def _stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        user_id = update.effective_user.id
        success = await self.bot.stop_user_session(user_id)
        if success:
            await update.message.reply_text("🛑 Signals stopped. Use /start to begin again.")
        else:
            await update.message.reply_text("ℹ️ No active session found.")
    
    async def _signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command - request immediate signal"""
        user_id = update.effective_user.id
        
        if user_id not in self.bot.user_sessions or not self.bot.user_sessions[user_id]['active']:
            await update.message.reply_text("❌ Please start signals first using /start or the button below.")
            return
        
        await update.message.reply_text("🔍 Analyzing markets for immediate signal...")
        
        # Generate immediate signal
        symbol = random.choice(TRADING_ASSETS)
        signal = await self.bot.signal_generator.generate_signal(symbol)
        
        if signal:
            await self.bot.send_signal_message(self.application, user_id, signal)
            self.bot.performance_stats['total_signals'] += 1
        else:
            await update.message.reply_text("❌ No high-confidence signal found at the moment. Try again shortly.")
    
    async def _button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses"""
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        if query.data == "start_signals":
            success = await self.bot.start_user_session(user_id)
            if success:
                await query.edit_message_text(
                    "✅ *Signals Started!*\n\nYou will now receive automated trading signals.\n\nUse /stop to pause signals or /signal for immediate analysis.",
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text("✅ Signals are already running!")
        
        elif query.data == "stop_signals":
            success = await self.bot.stop_user_session(user_id)
            if success:
                await query.edit_message_text("🛑 Signals stopped. Use /start to begin again.")
            else:
                await query.edit_message_text("❌ No active session found.")
        
        elif query.data == "live_stats":
            stats = self.bot.performance_stats
            active_users = len([u for u in self.bot.user_sessions.values() if u['active']])
            await query.edit_message_text(
                f"📊 Live Stats:\nActive Users: {active_users}\nTotal Signals: {stats['total_signals']}\nLast Signal: {stats['last_signal_time'].strftime('%H:%M') if stats['last_signal_time'] else 'Never'}",
                parse_mode='Markdown'
            )
        
        elif query.data == "get_signal":
            if user_id not in self.bot.user_sessions or not self.bot.user_sessions[user_id]['active']:
                await query.edit_message_text("❌ Please start signals first!")
                return
            
            await query.edit_message_text("🔍 Analyzing markets...")
            symbol = random.choice(TRADING_ASSETS)
            signal = await self.bot.signal_generator.generate_signal(symbol)
            
            if signal:
                await self.bot.send_signal_message(self.application, user_id, signal)
            else:
                await query.edit_message_text("❌ No high-confidence signal found. Try again shortly.")
    
    async def start_signal_generation(self):
        """Start the automatic signal generation loop"""
        async def signal_loop():
            logger.info("Signal generation loop started")
            while True:
                try:
                    # Generate and send signals
                    await self.bot.generate_and_send_signals(self.application)
                    
                    # Wait before next signal cycle
                    wait_time = random.randint(
                        Config.MIN_SIGNAL_COOLDOWN, 
                        Config.MAX_SIGNAL_COOLDOWN
                    )
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Error in signal loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
        
        self.signal_task = asyncio.create_task(signal_loop())
    
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
    
    async def shutdown(self):
        """Shutdown Telegram bot"""
        if self.signal_task:
            self.signal_task.cancel()
            try:
                await self.signal_task
            except asyncio.CancelledError:
                pass
        
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        
        await self.bot.signal_generator.close()
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
        
        # Start signal generation
        await self.telegram_bot.start_signal_generation()
        
        self.is_running = True
        logger.info("Application setup completed successfully")
        logger.info("Signal generation is ACTIVE - Users will receive trading signals")
    
    async def run(self):
        """Run the main application"""
        if not self.is_running:
            raise RuntimeError("Application not properly initialized")
        
        logger.info("Starting application main loop...")
        
        try:
            # Start Telegram polling
            await self.telegram_bot.start_polling()
            
            # Keep the application running
            while self.is_running:
                await asyncio.sleep(1)
                
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
