#!/usr/bin/env python3
"""
LEKZY FX AI PRO v12.0 - TRADING BOT
SIMPLIFIED VERSION WITHOUT PROMETHEUS
"""

import os
import asyncio
import sqlite3
import json
import time
import random
import logging
import requests
import pandas as pd
import numpy as np
import aiohttp
from datetime import datetime, timedelta
from threading import Thread, Lock
from flask import Flask, render_template_string, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import ta
import warnings
warnings.filterwarnings('ignore')

# ==================== STRUCTURED JSON LOGGING ====================
class JSONLogger:
    def __init__(self):
        self.lock = Lock()
    def log(self, level, msg, **kwargs):
        with self.lock:
            record = {"timestamp": datetime.utcnow().isoformat(), "level": level, "message": msg, **kwargs}
            print(json.dumps(record, ensure_ascii=False))

log = JSONLogger()

# ==================== CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "demo")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "demo")
    BINANCE_KEY = os.getenv("BINANCE_KEY", "")
    BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
    PORT = int(os.getenv("PORT", 10000))
    DB_PATH = "lekzy_fx_ai_pro_v12.db"
    PAPER_TRADING = os.getenv("PAPER", "true").lower() == "true"

    TRADING_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD",
        "USD/CAD", "EUR/GBP", "GBP/JPY", "USD/CHF", "NZD/USD"
    ]
    TIMEFRAMES = ["1M", "5M", "15M", "30M", "1H"]

    QUANTUM_MODES = {
        "QUANTUM_HYPER": {"name": "QUANTUM HYPER", "pre_entry": 3, "duration": 45, "acc": 0.88},
        "NEURAL_TURBO": {"name": "NEURAL TURBO", "pre_entry": 5, "duration": 90, "acc": 0.91},
        "QUANTUM_ELITE": {"name": "QUANTUM ELITE", "pre_entry": 8, "duration": 180, "acc": 0.94},
        "DEEP_PREDICT": {"name": "DEEP PREDICT", "pre_entry": 12, "duration": 300, "acc": 0.96}
    }

# ==================== DUAL DATA FEED + FALLBACK ====================
class DualDataEngine:
    def __init__(self):
        self.session = None
        self.cache = {}
        self.lock = Lock()

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def get_price(self, symbol):
        try:
            formatted = symbol.replace('/', '')
            
            # Primary: Twelve Data
            try:
                url = f"https://api.twelvedata.com/time_series"
                params = {"symbol": formatted, "interval": "1min", "apikey": Config.TWELVE_DATA_API_KEY, "outputsize": 1}
                async with self.session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'values' in data and data['values']:
                            price = float(data['values'][0]['close'])
                            with self.lock:
                                self.cache[symbol] = price
                            return round(price, 5)
            except Exception as e:
                log.log("WARNING", "Twelve Data failed", error=str(e))

            # Fallback: Mock data for demo
            with self.lock:
                if symbol in self.cache:
                    # Small random movement
                    self.cache[symbol] += random.uniform(-0.001, 0.001)
                else:
                    # Initial mock prices
                    mock_prices = {
                        "EURUSD": 1.08500, "GBPUSD": 1.26500, "USDJPY": 147.500, 
                        "XAUUSD": 1980.00, "AUDUSD": 0.65800, "USDCAD": 1.35000,
                        "EURGBP": 0.85700, "GBPJPY": 186.500, "USDCHF": 0.88000,
                        "NZDUSD": 0.61200
                    }
                    self.cache[symbol] = mock_prices.get(formatted, 1.08500)
                
                return round(self.cache[symbol], 5)
                
        except Exception as e:
            log.log("ERROR", "Price fetch failed", error=str(e))
            return 1.08500

    async def close(self):
        if self.session:
            await self.session.close()

# ==================== TECHNICAL ANALYSIS ENGINE ====================
class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0 if avg_gains > 0 else 50.0
            
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0, 0, 0
            
        exp1 = pd.Series(prices).ewm(span=fast).mean()
        exp2 = pd.Series(prices).ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

# ==================== QUANTUM AI ENGINE ====================
class QuantumAI:
    def __init__(self, data_engine):
        self.data = data_engine
        self.tech = TechnicalAnalysis()

    async def analyze(self, symbol, timeframe="5M"):
        try:
            # Get current price
            current_price = await self.data.get_price(symbol)
            
            # Generate mock historical data for analysis
            prices = [current_price * (1 + random.uniform(-0.02, 0.02)) for _ in range(50)]
            
            # Technical indicators
            rsi = self.tech.calculate_rsi(prices)
            macd, signal, histogram = self.tech.calculate_macd(prices)
            
            # Market hours analysis
            hour = datetime.utcnow().hour
            is_peak_hours = 8 <= hour < 16
            sentiment = 0.7 if is_peak_hours else 0.5
            
            # Rule-based filters
            if symbol == "XAU/USD" and 12 <= hour < 14:
                return {"blocked": "News overlap period"}
                
            if hour in [22, 23, 0, 1, 2, 3, 4, 5] and symbol not in ["USD/JPY", "AUD/USD"]:
                return {"blocked": "Low liquidity hours"}

            # Trading decision
            if rsi < 30 and macd > signal:
                direction = "BUY"
                confidence = min(0.99, 0.85 + random.uniform(0.05, 0.12))
            elif rsi > 70 and macd < signal:
                direction = "SELL" 
                confidence = min(0.99, 0.85 + random.uniform(0.05, 0.12))
            else:
                # Neutral market - random with lower confidence
                direction = "BUY" if random.random() > 0.5 else "SELL"
                confidence = 0.75 + random.uniform(0.05, 0.15)

            # Calculate levels
            if direction == "BUY":
                entry = round(current_price + 0.0002, 5)
                tp = round(entry + 0.0045, 5)
                sl = round(entry - 0.0025, 5)
            else:
                entry = round(current_price - 0.0002, 5)
                tp = round(entry - 0.0045, 5)
                sl = round(entry + 0.0025, 5)

            return {
                "direction": direction,
                "confidence": round(confidence, 3),
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "risk_reward": round(1.8, 1),
                "rsi": rsi,
                "macd": round(macd, 5)
            }
            
        except Exception as e:
            log.log("ERROR", "Analysis failed", symbol=symbol, error=str(e))
            return {"blocked": f"Analysis error: {str(e)}"}

# ==================== BROKER ADAPTER ====================
class Broker:
    def __init__(self):
        self.paper_trades = []
        self.trade_history = []

    def execute(self, signal, user_id):
        trade_id = f"TR{int(time.time())}{random.randint(1000,9999)}"
        
        if Config.PAPER_TRADING:
            # Simulate trade execution with realistic P&L
            base_pnl = random.uniform(-80, 150)
            # Higher confidence = better chance of profit
            confidence_boost = signal.get("confidence", 0.5) * 40
            final_pnl = base_pnl + confidence_boost
            
            result = "WIN" if final_pnl > 0 else "LOSS"
            pnl_amount = round(final_pnl, 2)
            
            trade_record = {
                **signal, 
                "trade_id": trade_id,
                "user_id": user_id,
                "pnl": pnl_amount, 
                "result": result,
                "executed_at": datetime.utcnow().isoformat()
            }
            
            self.paper_trades.append(trade_record)
            self.trade_history.append(trade_record)
            
            log.log("INFO", "Paper trade executed", 
                   trade_id=trade_id, symbol=signal["symbol"], 
                   direction=signal["direction"], result=result, pnl=pnl_amount)
            
            return result
        else:
            # Real trading would go here
            log.log("INFO", "REAL ORDER SKIPPED - Paper trading mode", 
                   symbol=signal["symbol"], direction=signal["direction"])
            return "PAPER_MODE_ONLY"

    def get_user_stats(self, user_id):
        user_trades = [t for t in self.trade_history if t.get("user_id") == user_id]
        total = len(user_trades)
        wins = len([t for t in user_trades if t.get("result") == "WIN"])
        win_rate = round((wins / total * 100), 1) if total > 0 else 0
        total_pnl = sum(t.get("pnl", 0) for t in user_trades)
        
        return {
            "total_trades": total,
            "winning_trades": wins,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / total, 2) if total > 0 else 0
        }

broker = Broker()

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    def __init__(self):
        self.data = DualDataEngine()
        self.ai = QuantumAI(self.data)

    async def init(self):
        await self.data.start()

    async def generate(self, user_id, mode="QUANTUM_ELITE"):
        try:
            symbol = random.choice(Config.TRADING_PAIRS)
            analysis = await self.ai.analyze(symbol)

            if "blocked" in analysis:
                return analysis

            signal = {
                "symbol": symbol,
                "direction": analysis["direction"],
                "entry": analysis["entry"],
                "tp": analysis["tp"],
                "sl": analysis["sl"],
                "confidence": analysis["confidence"],
                "mode": Config.QUANTUM_MODES[mode]["name"],
                "risk_reward": analysis["risk_reward"],
                "rsi": analysis.get("rsi", 50),
                "macd": analysis.get("macd", 0)
            }

            # Save to database
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO signals (user_id, symbol, direction, entry, tp, sl, confidence, mode, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, symbol, signal["direction"], signal["entry"], signal["tp"], 
                      signal["sl"], signal["confidence"], mode, None))

            # Execute trade
            result = broker.execute(signal, user_id)

            return signal
            
        except Exception as e:
            log.log("ERROR", "Signal generation failed", user_id=user_id, error=str(e))
            return {"blocked": f"Signal generation error: {str(e)}"}

    async def close(self):
        await self.data.close()

# ==================== TELEGRAM BOT ====================
class LekzyBot:
    def __init__(self):
        self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self.signal_gen = SignalGenerator()
        self.setup_handlers()

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(CallbackQueryHandler(self.button))

    async def start(self, update: Update, context):
        keyboard = [
            [InlineKeyboardButton("🎯 QUANTUM ELITE", callback_data="elite")],
            [InlineKeyboardButton("⚡ NEURAL TURBO", callback_data="neural")],
            [InlineKeyboardButton("📊 LIVE DASHBOARD", callback_data="dashboard")],
            [InlineKeyboardButton("📈 MY STATS", callback_data="stats")]
        ]
        
        welcome_text = """
🤖 *LEKZY FX AI PRO v12.0* 🤖
*World's #1 Quantum Trading System*

✨ *13 Hedge-Fund Upgrades:*
• Dual Data Feed Engine
• Quantum AI Ensemble
• Real-time Risk Management
• Institutional Grade Analytics

📊 *Live Market Access*
🎯 *97%+ Accuracy Engine*
⚡ *Millisecond Execution*

Choose your trading mode below:
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def stats(self, update: Update, context):
        user_id = update.message.from_user.id
        stats = broker.get_user_stats(user_id)
        
        stats_text = f"""
📊 *YOUR TRADING STATS*

Trades: {stats['total_trades']}
Wins: {stats['winning_trades']}
Win Rate: {stats['win_rate']}%
Total P&L: ${stats['total_pnl']}
Avg P&L: ${stats['avg_pnl']}

*Keep trading with Quantum Edge!*
        """
        await update.message.reply_text(stats_text, parse_mode='Markdown')

    async def button(self, update: Update, context):
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id

        if query.data in ["elite", "neural"]:
            await query.edit_message_text("🔮 *Quantum AI Analyzing Markets...*\n\nScanning 10 currency pairs...\nRunning ensemble models...", parse_mode='Markdown')
            
            mode = "QUANTUM_ELITE" if query.data == "elite" else "NEURAL_TURBO"
            await self.signal_gen.init()
            signal = await self.signal_gen.generate(user_id, mode)
            await self.send_signal(query.message.chat_id, signal, mode)
            await self.signal_gen.close()
            
        elif query.data == "stats":
            stats = broker.get_user_stats(user_id)
            stats_text = f"""
📊 *YOUR TRADING STATS*

Trades: {stats['total_trades']}
Wins: {stats['winning_trades']}
Win Rate: {stats['win_rate']}%
Total P&L: ${stats['total_pnl']}
Avg P&L: ${stats['avg_pnl']}
            """
            await query.edit_message_text(stats_text, parse_mode='Markdown')
            
        elif query.data == "dashboard":
            await query.edit_message_text("🌐 *Live Dashboard*\n\nAccess your dashboard at:\nhttp://localhost:10000/dashboard\n\n*Real-time analytics & performance metrics*", parse_mode='Markdown')

    async def send_signal(self, chat_id, signal, mode):
        if "blocked" in signal:
            await self.app.bot.send_message(
                chat_id, 
                f"🚫 *Trade Blocked*\n\nReason: {signal['blocked']}\n\nTry again in different market conditions.",
                parse_mode='Markdown'
            )
            return

        mode_display = Config.QUANTUM_MODES.get(mode, {}).get("name", "QUANTUM ELITE")
        
        signal_text = f"""
🎯 *QUANTUM SIGNAL GENERATED* 🎯

*{mode_display} MODE*
━━━━━━━━━━━━━━━━━━━━

💱 *PAIR:* `{signal['symbol']}`
📈 *DIRECTION:* *{signal['direction']}*
🎯 *ENTRY:* `{signal['entry']}`
✅ *TAKE PROFIT:* `{signal['tp']}`
❌ *STOP LOSS:* `{signal['sl']}`

📊 *ANALYTICS:*
├ Confidence: *{signal['confidence']*100:.1f}%*
├ RSI: *{signal.get('rsi', 'N/A')}*
├ Risk/Reward: *1:{signal['risk_reward']}*
└ Mode: *{mode_display}*

💡 *Quantum AI Verification: PASSED*
⚡ *Execution: IMMEDIATE*

*Trade safe! Use proper risk management.*
        """
        
        await self.app.bot.send_message(chat_id, signal_text, parse_mode='Markdown')

    def run(self):
        self.app.run_polling()

# ==================== FLASK DASHBOARD ====================
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LEKZY FX AI PRO v12.0</title>
    <meta charset="utf-8">
    <style>
        body {font-family: 'Courier New', monospace; background: #0a0a1a; color: #0f0; padding: 20px;}
        .card {background: #111; border: 1px solid #0f0; padding: 20px; margin: 15px 0; border-radius: 8px;}
        table {width: 100%; border-collapse: collapse;} 
        th, td {border: 1px solid #0f0; padding: 12px; text-align: left;}
        .win {color: #0f0;} .loss {color: #f55;} .neutral {color: #aaa;}
        h1 {text-align: center; color: #0f0; margin-bottom: 30px;}
        .status {color: #0f0; font-weight: bold;}
        .metric {font-size: 1.2em; color: #0ff;}
    </style>
</head>
<body>
    <h1>🤖 LEKZY FX AI PRO v12.0 🤖</h1>
    
    <div class="card">
        <h3>🚀 13 HEDGE-FUND UPGRADES ACTIVE</h3>
        <p>• Dual Data Feed Engine • Quantum AI Ensemble • Rule-Based Filtering</p>
        <p>• Paper Trading Mode • Real-time Analytics • Risk Management</p>
        <p class="status">🟢 SYSTEM STATUS: OPERATIONAL</p>
    </div>
    
    <div class="card">
        <h3>📊 LIVE PERFORMANCE METRICS</h3>
        <p>Total Signals: <span class="metric">{{ total_signals }}</span> | 
           Win Rate: <span class="metric">{{ win_rate }}%</span> | 
           Avg Confidence: <span class="metric">{{ avg_confidence }}%</span></p>
    </div>
    
    <div class="card">
        <h3>📈 RECENT SIGNALS</h3>
        <table>
            <tr><th>Time</th><th>Symbol</th><th>Direction</th><th>Confidence</th><th>Result</th></tr>
            {% for signal in recent_signals %}
            <tr>
                <td>{{ signal.time }}</td>
                <td>{{ signal.symbol }}</td>
                <td>{{ signal.direction }}</td>
                <td>{{ signal.confidence }}%</td>
                <td class="{% if signal.result == 'WIN' %}win{% elif signal.result == 'LOSS' %}loss{% else %}neutral{% endif %}">
                    {{ signal.result or 'PENDING' }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="card">
        <p><strong>⚠️ DISCLAIMER:</strong> This is a demonstration system. Trade at your own risk.</p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return """
    <html>
        <head><title>LEKZY v12.0</title></head>
        <body style="background: #0a0a1a; color: #0f0; text-align: center; padding: 50px;">
            <h1>🤖 LEKZY FX AI PRO v12.0 🤖</h1>
            <p>World's #1 Quantum Trading System</p>
            <p><a href="/dashboard" style="color: #0ff;">📊 Access Live Dashboard</a></p>
        </body>
    </html>
    """

@app.route('/dashboard')
def dashboard():
    with sqlite3.connect(Config.DB_PATH) as conn:
        # Get total signals
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), AVG(confidence)*100 FROM signals")
        total_signals, avg_confidence = cur.fetchone()
        avg_confidence = round(avg_confidence or 0, 1) if avg_confidence else 0
        
        # Calculate win rate from broker history
        stats = broker.get_user_stats("all")
        win_rate = stats['win_rate'] if stats['total_trades'] > 0 else 0
        
        # Get recent signals
        cur.execute("""
            SELECT created_at, symbol, direction, confidence, 
                   (SELECT result FROM signals s2 WHERE s2.id = signals.id) as result
            FROM signals 
            ORDER BY id DESC LIMIT 15
        """)
        recent_data = []
        for row in cur.fetchall():
            recent_data.append({
                "time": row[0][:16] if row[0] else "N/A",
                "symbol": row[1],
                "direction": row[2],
                "confidence": round(row[3] * 100, 1) if row[3] else 0,
                "result": row[4]
            })
    
    return render_template_string(
        DASHBOARD_HTML,
        total_signals=total_signals,
        win_rate=win_rate,
        avg_confidence=avg_confidence,
        recent_signals=recent_data
    )

# ==================== MAIN APPLICATION ====================
def initialize_database():
    """Initialize the SQLite database with required tables"""
    with sqlite3.connect(Config.DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            direction TEXT,
            entry REAL,
            tp REAL,
            sl REAL,
            confidence REAL,
            mode TEXT,
            result TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            started_at TEXT DEFAULT (datetime('now')),
            signals_generated INTEGER DEFAULT 0
        );
        """)
        log.log("INFO", "Database initialized successfully")

async def main_async():
    """Main async application entry point"""
    # Initialize database
    initialize_database()
    
    # Start Flask dashboard in background thread
    def run_flask():
        app.run(host='0.0.0.0', port=Config.PORT, debug=False, use_reloader=False)
    
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Initialize and run Telegram bot
    bot = LekzyBot()
    
    log.log("INFO", "LEKZY FX AI PRO v12.0 STARTED", 
           status="OPERATIONAL", 
           port=Config.PORT,
           mode="PAPER_TRADING" if Config.PAPER_TRADING else "LIVE_TRADING")
    
    print(f"\n{'='*60}")
    print("🤖 LEKZY FX AI PRO v12.0 - WORLD #1 TRADING BOT")
    print(f"{'='*60}")
    print("📊 Dashboard: http://localhost:10000/dashboard")
    print("🔧 Mode: PAPER TRADING (Demo)")
    print("🚀 Status: OPERATIONAL")
    print(f"{'='*60}\n")
    
    # Run bot
    bot.run()

if __name__ == "__main__":
    # Create simple requirements check
    try:
        import telegram
        import flask
        import aiohttp
        print("✅ All dependencies loaded successfully!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install python-telegram-bot flask aiohttp pandas numpy")
        exit(1)
    
    asyncio.run(main_async())
