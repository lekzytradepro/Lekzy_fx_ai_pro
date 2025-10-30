import os
import asyncio
import aiohttp
import sqlite3
import json
import time
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
import joblib
import pytz

# Telegram imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ML libs
XGB_AVAILABLE = False
SKLEARN_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------- Config & environment --------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "")  # optional
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # required for admin actions
DB_PATH = os.getenv("DB_PATH", "trade_data.db")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
SCALER_DIR = os.getenv("SCALER_DIR", "scalers")
RETRAIN_CANDLES = int(os.getenv("RETRAIN_CANDLES", "200"))
PREENTRY_DEFAULT = int(os.getenv("PREENTRY_DEFAULT", "30"))
HTTP_PORT = int(os.getenv("PORT", 8080))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is required in environment variables")

if not ADMIN_TOKEN:
    print("⚠️  ADMIN_TOKEN is not set. Admin-only commands will be blocked until set.")

# timezone: UTC+1
TZ = pytz.timezone("Etc/GMT-1")  # UTC+1

# ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# -------------------- Tunables --------------------
MIN_SIGNAL_COOLDOWN = 40
MAX_SIGNAL_COOLDOWN = 180
CANDLE_LIMIT = 500
TWELVE_CACHE_TTL = 15  # seconds
# TwelveData free tier ~ 8 requests/min - we'll rate limit safely
TD_RATE_LIMIT_PER_MIN = 8

VOLATILITY_MIN_FOREX = 0.00025
VOLATILITY_MAX_FOREX = 0.006
VOLATILITY_MIN_CRYPTO = 12
VOLATILITY_MAX_CRYPTO = 800

DEFAULT_TIMEFRAME_MODE = "auto"  # 'auto' | '1m' | '5m'

# in-memory caches and trackers
_candle_cache: Dict[str, Dict[str, Any]] = {}
active_users = set()
user_start_times: Dict[int, float] = {}
user_settings: Dict[int, Dict[str, Any]] = {}  # e.g., {'preentry': 30, 'timeframe_mode': 'auto'}

# -------------------- Watchlists (stable pairs + crypto + commodities) --------------------
ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/JPY", "GBP/JPY",
    "USD/CAD", "EUR/GBP", "USD/CHF", "BTC/USD", "ETH/USD", "XAU/USD", "XAG/USD"
]

# -------------------- Simple HTTP health server --------------------
from http.server import HTTPServer, BaseHTTPRequestHandler
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b"Lekzy_FX_AI_Pro - Live")
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, format, *args):
        pass

def start_http_server():
    try:
        server = HTTPServer(("0.0.0.0", HTTP_PORT), HealthHandler)
        print(f"HTTP health server listening on {HTTP_PORT}")
        server.serve_forever()
    except Exception as e:
        print("HTTP server failed:", e)

# -------------------- DB --------------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT,
            symbol TEXT,
            side TEXT,
            timeframe TEXT,
            entry_price REAL,
            confidence REAL,
            label INTEGER,
            details TEXT,
            timestamp TEXT,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_time TEXT,
            profit REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS authorized_users (
            chat_id INTEGER PRIMARY KEY,
            username TEXT,
            authorized_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            chat_id INTEGER PRIMARY KEY,
            username TEXT,
            status TEXT DEFAULT 'pending',
            requested_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Rate limiter for TwelveData (free tier safe) --------------------
_td_call_timestamps: List[float] = []  # epoch seconds of recent calls
_td_lock = asyncio.Lock()

async def ensure_twelvedata_rate_limit():
    """Ensure no more than TD_RATE_LIMIT_PER_MIN calls per 60-second window."""
    async with _td_lock:
        now = time.time()
        # remove timestamps older than 60s
        cutoff = now - 60.0
        while _td_call_timestamps and _td_call_timestamps[0] < cutoff:
            _td_call_timestamps.pop(0)
        if len(_td_call_timestamps) >= TD_RATE_LIMIT_PER_MIN:
            # sleep until the oldest falls out of window
            wait = 60.0 - (now - _td_call_timestamps[0]) + 0.5
            await asyncio.sleep(wait)
            # cleanup after waiting
            now2 = time.time()
            cutoff2 = now2 - 60.0
            while _td_call_timestamps and _td_call_timestamps[0] < cutoff2:
                _td_call_timestamps.pop(0)
        _td_call_timestamps.append(time.time())

# -------------------- Utilities & indicators --------------------
def fname_for_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(".", "_").replace("-", "_")

def model_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"{fname_for_symbol(symbol)}_model.pkl")

def scaler_path(symbol: str) -> str:
    return os.path.join(SCALER_DIR, f"{fname_for_symbol(symbol)}_scaler.pkl")

def sma(arr: np.ndarray, period: int) -> float:
    if len(arr) == 0: return 0.0
    if len(arr) < period: return float(np.mean(arr))
    return float(np.mean(arr[-period:]))
def ema_array(arr: np.ndarray, period: int) -> float:
    if len(arr) == 0: return 0.0
    a = 2 / (period + 1)
    s = arr[0]
    for v in arr[1:]:
        s = a * v + (1 - a) * s
    return float(s)
def rsi_calc(closes: np.ndarray, period=14) -> float:
    if len(closes) < 2: return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else (np.mean(gains) if gains.size else 0.0)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else (np.mean(losses) if losses.size else 0.0)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / (avg_loss + 1e-9)
    return float(100 - (100 / (1 + rs)))
def bollinger_width(closes: np.ndarray, period=20) -> float:
    if len(closes) < 2: return 0.0
    sma_val = sma(closes, min(period, len(closes)))
    std = float(np.std(closes[-period:])) if len(closes) >= period else float(np.std(closes))
    upper = sma_val + 2 * std
    lower = sma_val - 2 * std
    return float((upper - lower) / (abs(sma_val) + 1e-9))
def atr_calc(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period=14) -> float:
    if len(closes) < 2: return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return float(np.mean(trs[-period:])) if trs else 0.0

# -------------------- Per-asset ML model --------------------
class PerAssetModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.mpath = model_path(symbol)
        self.spath = scaler_path(symbol)
        self.model = None
        self.scaler = None
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.mpath):
            try:
                self.model = joblib.load(self.mpath)
            except Exception:
                self.model = None
        if os.path.exists(self.spath) and SKLEARN_AVAILABLE:
            try:
                self.scaler = joblib.load(self.spath)
            except Exception:
                self.scaler = None

        if self.model is None:
            if XGB_AVAILABLE:
                self.model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric="logloss")
            elif SKLEARN_AVAILABLE:
                self.model = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
            else:
                self.model = None

        if self.scaler is None and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()

        try:
            if self.model is not None:
                joblib.dump(self.model, self.mpath)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.spath)
        except Exception:
            pass

    def features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vec = [
            float(features.get("last_return", 0.0)),
            float(features.get("volatility", 0.0)),
            float(features.get("ema_diff", 0.0)),
            float(features.get("rsi", 50.0)),
            float(features.get("macd", 0.0)),
            float(features.get("bb_width", 0.0)),
            float(features.get("atr", 0.0)),
            float(features.get("up_count_5", 0.0))
        ]
        return np.array(vec, dtype=float)

    def predict(self, features: Dict[str, Any]) -> (str, float):
        if self.model is None:
            return ("UP" if random.random() < 0.6 else "DOWN"), round(random.uniform(60,85),2)
        x = self.features_to_vector(features).reshape(1, -1)
        if self.scaler is not None:
            try:
                x = self.scaler.transform(x)
            except Exception:
                try:
                    self.scaler.partial_fit(x)
                    x = self.scaler.transform(x)
                except Exception:
                    pass
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
                up_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                direction = "UP" if up_prob >= 0.5 else "DOWN"
                confidence = round(max(up_prob, 1 - up_prob) * 100, 2)
                return direction, confidence
            else:
                pred = int(self.model.predict(x)[0])
                return ("UP" if pred == 1 else "DOWN"), 60.0
        except Exception:
            return ("UP" if random.random() < 0.6 else "DOWN"), round(random.uniform(60,85),2)

    def train(self, X: np.ndarray, y: np.ndarray, incremental: bool=False) -> bool:
        if self.model is None:
            return False
        if self.scaler is not None:
            try:
                if hasattr(self.scaler, "partial_fit"):
                    self.scaler.partial_fit(X)
                else:
                    self.scaler.fit(X)
                Xs = self.scaler.transform(X)
            except Exception:
                Xs = X
        else:
            Xs = X
        try:
            if XGB_AVAILABLE and isinstance(self.model, XGBClassifier):
                self.model.fit(Xs, y)
            elif SKLEARN_AVAILABLE and hasattr(self.model, "partial_fit") and incremental:
                if not hasattr(self.model, "classes_"):
                    self.model.partial_fit(Xs, y, classes=np.array([0,1]))
                else:
                    self.model.partial_fit(Xs, y)
            else:
                self.model.fit(Xs, y)
            joblib.dump(self.model, self.mpath)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.spath)
            return True
        except Exception as e:
            print("Train error:", e)
            return False

# -------------------- Core bot class --------------------
class LekzyFXAI:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.models: Dict[str, PerAssetModel] = {}
        self.total_signals_sent = 0
        self.user_loops: Dict[int, asyncio.Task] = {}
        self.retrain_task: Optional[asyncio.Task] = None

    async def init_session(self):
        if self.session and not self.session.closed:
            return
        self.session = aiohttp.ClientSession()
        print("HTTP session initialized")

    # --- auth helpers ---
    def authorize_user(self, chat_id:int, username:str):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO authorized_users (chat_id, username, authorized_at) VALUES (?, ?, ?)", (chat_id, username or "", datetime.now().isoformat()))
            conn.commit(); conn.close()
        except Exception as e:
            print("authorize error:", e)

    def deauthorize_user(self, chat_id:int):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("DELETE FROM authorized_users WHERE chat_id = ?", (chat_id,))
            conn.commit(); conn.close()
        except Exception as e:
            print("deauthorize error:", e)

    def is_authorized(self, chat_id:int) -> bool:
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT chat_id FROM authorized_users WHERE chat_id = ?", (chat_id,))
            r = cur.fetchone(); conn.close()
            return bool(r)
        except Exception as e:
            print("is_authorized error:", e)
            return False

    # --- subscriber flow (users can request subscription, admin approves) ---
    def request_subscribe(self, chat_id:int, username:str):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO subscribers (chat_id, username, status, requested_at) VALUES (?, ?, ?, ?)",
                        (chat_id, username or "", "pending", datetime.now().isoformat()))
            conn.commit(); conn.close()
            return True
        except Exception as e:
            print("subscribe error:", e)
            return False

    def approve_subscriber(self, chat_id:int):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("UPDATE subscribers SET status='approved' WHERE chat_id = ?", (chat_id,))
            conn.commit(); conn.close()
            # add to authorized_users
            self.authorize_user(chat_id, "")
            return True
        except Exception as e:
            print("approve error:", e)
            return False

    def list_pending_subscribers(self) -> List[Dict[str,Any]]:
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT chat_id, username, requested_at FROM subscribers WHERE status='pending'")
            rows = cur.fetchall(); conn.close()
            return [{"chat_id": r[0], "username": r[1], "requested_at": r[2]} for r in rows]
        except Exception as e:
            print("list pending error:", e)
            return []

    # -------------------- TwelveData fetch --------------------
    async def fetch_market_candles(self, symbol: str, interval: str = "1min", limit: int = 200) -> List[Dict[str,Any]]:
        now_ts = time.time()
        cache_key = f"{symbol}_{interval}"
        cache_entry = _candle_cache.get(cache_key)
        if cache_entry and now_ts - cache_entry['ts'] < TWELVE_CACHE_TTL:
            return cache_entry['data']

        # rate-limit guard
        if TWELVE_API_KEY:
            await ensure_twelvedata_rate_limit()

        if not self.session or self.session.closed:
            await self.init_session()

        if TWELVE_API_KEY:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": str(min(limit, CANDLE_LIMIT)),
                "format": "JSON",
                "apikey": TWELVE_API_KEY
            }
            try:
                async with self.session.get(url, params=params, timeout=15) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        print("TwelveData HTTP", resp.status, text)
                        raise Exception("TD error")
                    data = json.loads(text)
                    if "values" in data:
                        values = list(reversed(data["values"]))[:limit]
                        candles = []
                        for v in values:
                            candles.append({
                                "t": v.get("datetime") or v.get("datetime",""),
                                "open": float(v.get("open",0.0)),
                                "high": float(v.get("high",0.0)),
                                "low": float(v.get("low",0.0)),
                                "close": float(v.get("close",0.0)),
                                "volume": float(v.get("volume",0.0)) if v.get("volume") is not None else 0.0
                            })
                        _candle_cache[cache_key] = {"ts": now_ts, "data": candles}
                        return candles
                    else:
                        print("TwelveData missing values:", data)
            except Exception as e:
                print("fetch_market_candles TD error:", e)

        # fallback synthetic candles (useful for testing)
        candles = []
        price = random.uniform(1.0, 100.0)
        for i in range(limit):
            change = random.uniform(-0.002, 0.002)
            o = price
            c = price * (1 + change)
            h = max(o, c) * (1 + random.uniform(0, 0.001))
            l = min(o, c) * (1 - random.uniform(0, 0.001))
            v = random.uniform(10, 1000)
            candles.append({"t": (datetime.utcnow() - timedelta(minutes=limit - i)).isoformat(),
                            "open": o, "high": h, "low": l, "close": c, "volume": v})
            price = c
        _candle_cache[cache_key] = {"ts": now_ts, "data": candles}
        return candles

    # -------------------- Feature extraction --------------------
    def extract_features(self, symbol: str, candles: List[Dict[str,Any]]) -> Dict[str,Any]:
        if not candles or len(candles) < 6:
            return {"last_return":0,"volatility":0,"ema_diff":0,"rsi":50,"macd":0,"bb_width":0,"atr":0,"up_count_5":0}
        closes = np.array([c['close'] for c in candles], dtype=float)
        highs = np.array([c['high'] for c in candles], dtype=float)
        lows = np.array([c['low'] for c in candles], dtype=float)
        vols = np.array([c.get('volume',0.0) for c in candles], dtype=float)
        returns = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-12)
        last_return = float(returns[-1]) if len(returns) >= 1 else 0.0
        volatility = float(np.std(returns[-20:])) if len(returns) >= 2 else 0.0
        ema5 = ema_array(closes[-13:], 5) if len(closes) >= 5 else float(closes[-1])
        ema13 = ema_array(closes[-13:], 13) if len(closes) >= 13 else float(closes[-1])
        ema_diff = float((ema5 - ema13) / (abs(ema13) + 1e-9))
        macd = 0.0
        try:
            macd = float(np.mean(closes[-12:]) - np.mean(closes[-26:]))
        except:
            macd = 0.0
        rsi = rsi_calc(closes, 14)
        bb_width = bollinger_width(closes, 20)
        atr = atr_calc(highs, lows, closes, 14)
        up_count_5 = float(sum(1 for i in range(-5, 0) if i != 0 and closes[i] > closes[i-1]))
        return {
            "last_return": last_return,
            "volatility": volatility,
            "ema_diff": ema_diff,
            "rsi": float(rsi),
            "macd": float(macd),
            "bb_width": float(bb_width),
            "atr": float(atr),
            "up_count_5": up_count_5
        }

    def asset_volatility_ok(self, symbol: str, features: Dict[str, Any]) -> bool:
        name = symbol.upper()
        atr = features.get("atr", 0.0)
        if any(x in name for x in ["BTC","ETH","LTC","BNB","ADA","SOL"]):
            return VOLATILITY_MIN_CRYPTO <= atr <= VOLATILITY_MAX_CRYPTO
        return VOLATILITY_MIN_FOREX <= atr <= VOLATILITY_MAX_FOREX

    # -------------------- Decide timeframe --------------------
    def decide_timeframe(self, user_tf_mode: str, confidence: float) -> str:
        if user_tf_mode in ("1m","5m"):
            return user_tf_mode
        if confidence >= 85.0:
            return "1m"
        return "5m"

    # -------------------- Generate signal (pre-entry + entry) --------------------
    async def generate_and_send(self, user_id:int, application, symbol:str, user_settings_local:dict):
        try:
            if symbol not in self.models:
                self.models[symbol] = PerAssetModel(symbol)
            model = self.models[symbol]

            candles_1m = await self.fetch_market_candles(symbol, interval="1min", limit=60)
            features_1m = self.extract_features(symbol, candles_1m)
            candles_5m = await self.fetch_market_candles(symbol, interval="5min", limit=200)
            features_5m = self.extract_features(symbol, candles_5m)

            dir1, conf1 = model.predict(features_1m)
            dir5, conf5 = model.predict(features_5m)
            chosen_conf = max(conf1, conf5)
            user_tf_mode = user_settings_local.get("timeframe_mode", DEFAULT_TIMEFRAME_MODE)
            tf_choice = self.decide_timeframe(user_tf_mode, chosen_conf)
            interval = "1min" if tf_choice == "1m" else "5min"
            features = features_1m if interval == "1min" else features_5m
            direction, confidence = model.predict(features)
            confidence = float(confidence)

            if not self.asset_volatility_ok(symbol, features):
                print(f"Skipped {symbol}, volatility out of range")
                return

            next_candle_info = self.get_next_candle_time(interval)
            next_open_utc = next_candle_info["next_candle_time"]
            seconds_to_entry = (next_open_utc - datetime.now(timezone.utc)).total_seconds()
            if seconds_to_entry < 2:
                return
            preentry = user_settings_local.get("preentry", PREENTRY_DEFAULT)
            preentry_seconds = min(preentry, max(1, int(seconds_to_entry - 1)))

            now_local = datetime.now(timezone.utc).astimezone(TZ)
            entry_local = next_open_utc.astimezone(TZ)
            dir_emoji = "🟢 BUY" if direction=="UP" else "🔴 SELL"
            strength = "💎 STRONG" if confidence >= 80 else ("⚡ MEDIUM" if confidence >= 65 else "⚠️ LOW")

            pre_msg = (
                f"⏰ PRE-ENTRY ALERT\n\n"
                f"📌 Pair: {symbol}\n"
                f"📈 Signal: {dir_emoji}\n"
                f"⏱ Timeframe: {'1M' if interval=='1min' else '5M'}\n"
                f"🔎 Confidence: {confidence:.1f}% ({strength})\n"
                f"🧾 Detected: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)\n"
                f"🕒 Planned Entry (new candle open): {entry_local.strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)\n"
                f"⏳ Pre-entry in {int(preentry_seconds)}s\n"
            )
            await application.bot.send_message(chat_id=user_id, text=pre_msg)

            await asyncio.sleep(preentry_seconds)
            # wait remainder to exact open
            now_utc = datetime.now(timezone.utc)
            remain = (next_open_utc - now_utc).total_seconds()
            if remain > 0:
                await asyncio.sleep(remain)

            entry_local_now = datetime.now(timezone.utc).astimezone(TZ)
            entry_msg = (
                f"✅ ENTRY (on new candle open)\n\n"
                f"📌 Pair: {symbol}\n"
                f"📈 Signal: {dir_emoji}\n"
                f"⏱ Timeframe: {'1M' if interval=='1min' else '5M'}\n"
                f"🔎 Confidence: {confidence:.1f}%\n"
                f"🕒 Entry Time: {entry_local_now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)\n"
            )
            await application.bot.send_message(chat_id=user_id, text=entry_msg)

            signal_id = f"SIG-{random.randint(1000,9999)}"
            try:
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO signals (signal_id, symbol, side, timeframe, entry_price, confidence, details, timestamp, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_id, symbol, "BUY" if direction=="UP" else "SELL",
                    '1m' if interval=='1min' else '5m', None, confidence,
                    json.dumps({"features": features}), datetime.utcnow().isoformat(), "OPEN"
                ))
                conn.commit(); conn.close()
            except Exception as e:
                print("DB insert failed:", e)

            self.total_signals_sent += 1
        except Exception as e:
            print("generate_and_send error:", e)

    def get_next_candle_time(self, interval: str) -> Dict[str, datetime]:
        now = datetime.now(timezone.utc)
        if interval in ("1min","1m"):
            nxt = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        else:
            minute = (now.minute // 5) * 5
            base = now.replace(minute=minute, second=0, microsecond=0)
            if now >= base + timedelta(minutes=5):
                base = base + timedelta(minutes=5)
            nxt = base + timedelta(minutes=5)
        return {"next_candle_time": nxt}

    # user loop
    async def start_for_user(self, user_id:int, application):
        if user_id in active_users:
            return "❗ Signals already running for you."
        if not self.is_authorized(user_id):
            return "🔒 You are not authorized. Use /subscribe to request access."
        user_settings.setdefault(user_id, {"preentry": PREENTRY_DEFAULT, "timeframe_mode": DEFAULT_TIMEFRAME_MODE})
        active_users.add(user_id)
        user_start_times[user_id] = time.time()
        task = asyncio.create_task(self._user_signal_loop(user_id, application))
        self.user_loops[user_id] = task
        if self.retrain_task is None:
            self.retrain_task = asyncio.create_task(self.periodic_retrain_loop())
        return "✅ Signals started. You will receive pre-entry + entry alerts."

    async def stop_for_user(self, user_id:int):
        if user_id not in active_users:
            return "ℹ️ Signals not running."
        if not self.is_authorized(user_id):
            return "🔒 Only authorized users can stop signals."
        active_users.discard(user_id)
        if user_id in user_start_times:
            del user_start_times[user_id]
        if user_id in self.user_loops:
            t = self.user_loops[user_id]
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            del self.user_loops[user_id]
        return "🛑 Signals stopped."

    async def _user_signal_loop(self, user_id:int, application):
        await asyncio.sleep(1)
        await application.bot.send_message(chat_id=user_id, text="🎯 Lekzy_FX_AI_Pro — Signals ACTIVATED")
        while user_id in active_users:
            try:
                symbol = random.choice(ASSETS)
                settings_local = user_settings.get(user_id, {"preentry": PREENTRY_DEFAULT, "timeframe_mode": DEFAULT_TIMEFRAME_MODE})
                await self.generate_and_send(user_id, application, symbol, settings_local)
                wait_t = random.randint(MIN_SIGNAL_COOLDOWN, MAX_SIGNAL_COOLDOWN)
                for _ in range(wait_t):
                    if user_id not in active_users:
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                print("user loop error:", e)
                await asyncio.sleep(5)

    # record exit and partial training
    def record_trade_exit(self, signal_id: str, exit_price: float, exit_time: Optional[str] = None) -> bool:
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT id, entry_price, details, status, symbol FROM signals WHERE signal_id = ?", (signal_id,))
            row = cur.fetchone()
            if not row:
                conn.close(); return False
            db_id, entry_price, details_json, status, symbol = row
            if status != "OPEN":
                conn.close(); return False
            profit = None
            try:
                if entry_price is not None:
                    profit = float(exit_price) - float(entry_price)
            except:
                profit = None
            label = 1 if profit is not None and profit > 0 else 0
            exit_time = exit_time or datetime.utcnow().isoformat()
            cur.execute("""
                UPDATE signals SET exit_price = ?, exit_time = ?, profit = ?, status = 'CLOSED', label = ?
                WHERE signal_id = ?
            """, (exit_price, exit_time, profit, label, signal_id))
            conn.commit()

            try:
                details = json.loads(details_json) if details_json else {}
                feats = details.get("features")
                if feats:
                    if symbol not in self.models:
                        self.models[symbol] = PerAssetModel(symbol)
                    m = self.models[symbol]
                    x = m.features_to_vector(feats).reshape(1, -1)
                    if m.scaler is not None:
                        try:
                            m.scaler.partial_fit(x)
                            xs = m.scaler.transform(x)
                        except Exception:
                            try:
                                m.scaler.fit(x); xs = m.scaler.transform(x)
                            except Exception:
                                xs = x
                    else:
                        xs = x
                    if SKLEARN_AVAILABLE and hasattr(m.model, "partial_fit"):
                        if not hasattr(m.model, "classes_"):
                            m.model.partial_fit(xs, np.array([label]), classes=np.array([0,1]))
                        else:
                            m.model.partial_fit(xs, np.array([label]))
                        joblib.dump(m.model, m.mpath)
            except Exception as e:
                print("partial training error:", e)

            conn.close()
            return True
        except Exception as e:
            print("record_trade_exit error:", e)
            return False

    # periodic retrain every 6 hours
    async def periodic_retrain_loop(self):
        while True:
            try:
                print("Periodic retrain starting...")
                for symbol in ASSETS:
                    try:
                        candles = await self.fetch_market_candles(symbol, interval="5min", limit=RETRAIN_CANDLES + 50)
                        if not candles or len(candles) < 60:
                            continue
                        closes = np.array([c['close'] for c in candles], dtype=float)
                        X_list = []; y_list = []
                        for i in range(30, len(closes)-1):
                            window = candles[max(0, i-120):i+1]
                            feats = self.extract_features(symbol, window)
                            X_list.append(self.vector_from_feats(feats))
                            y_list.append(1 if closes[i+1] > closes[i] else 0)
                        if not X_list:
                            continue
                        X = np.vstack(X_list); y = np.array(y_list)
                        if symbol not in self.models:
                            self.models[symbol] = PerAssetModel(symbol)
                        trained = self.models[symbol].train(X, y, incremental=SKLEARN_AVAILABLE)
                        print(f"Retrained {symbol}: success={trained}, samples={len(y)}")
                    except Exception as e:
                        print("retrain per-symbol error:", symbol, e)
                    await asyncio.sleep(0.8)
            except Exception as e:
                print("periodic retrain outer error:", e)
            # 6 hours sleep
            await asyncio.sleep(60 * 60 * 6)

    def vector_from_feats(self, feats: Dict[str,Any]) -> np.ndarray:
        return np.array([
            float(feats.get("last_return", 0.0)),
            float(feats.get("volatility", 0.0)),
            float(feats.get("ema_diff", 0.0)),
            float(feats.get("rsi", 50.0)),
            float(feats.get("macd", 0.0)),
            float(feats.get("bb_width", 0.0)),
            float(feats.get("atr", 0.0)),
            float(feats.get("up_count_5", 0.0))
        ], dtype=float)

# instantiate logic
lekzy = LekzyFXAI()

# -------------------- Telegram handlers & setup --------------------
def get_main_keyboard():
    keyboard = [
        [InlineKeyboardButton("🚀 START SIGNALS", callback_data="start_signals")],
        [InlineKeyboardButton("🛑 STOP SIGNALS", callback_data="stop_signals")],
        [InlineKeyboardButton("📊 LIVE STATS", callback_data="live_stats"),
         InlineKeyboardButton("🎯 ASSETS", callback_data="show_assets")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    is_auth = lekzy.is_authorized(user.id)
    welcome = (f"🎯 *Lekzy_FX_AI_Pro*\n\nAuthorized: {'✅' if is_auth else '❌'}\n"
               "Use /login <token> (admin) or /subscribe to request access. Then press START SIGNALS.")
    await update.message.reply_text(welcome, reply_markup=get_main_keyboard(), parse_mode='Markdown')

async def callback_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    try:
        if query.data == "start_signals":
            res = await lekzy.start_for_user(uid, context.application)
            await query.edit_message_text(res, reply_markup=get_main_keyboard())
        elif query.data == "stop_signals":
            res = await lekzy.stop_for_user(uid)
            await query.edit_message_text(res, reply_markup=get_main_keyboard())
        elif query.data == "live_stats":
            await show_stats(query)
        elif query.data == "show_assets":
            await show_assets(query)
    except Exception as e:
        print("callback error:", e)
        try:
            await query.edit_message_text("Error handling action.")
        except:
            pass

async def show_stats(query):
    uid = query.from_user.id
    running = uid in active_users
    rt = 0
    if uid in user_start_times:
        rt = time.time() - user_start_times[uid]
    text = (f"📊 Live Stats\nTotal signals: {lekzy.total_signals_sent}\n"
            f"Running: {'✅' if running else '❌'}\nRunning time: {int(rt/60)} min\nAssets monitored: {len(ASSETS)}")
    await query.edit_message_text(text, reply_markup=get_main_keyboard())

async def show_assets(query):
    text = f"Assets: {', '.join(ASSETS)}"
    await query.edit_message_text(text, reply_markup=get_main_keyboard())

# admin & auth commands
async def login_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    args = context.args
    if not ADMIN_TOKEN:
        await update.message.reply_text("Admin token not configured on server.")
        return
    if not args:
        await update.message.reply_text("Usage: /login <token>")
        return
    token = args[0].strip()
    if token != ADMIN_TOKEN:
        await update.message.reply_text("❌ Invalid token.")
        return
    lekzy.authorize_user(user.id, user.username or "")
    await update.message.reply_text("✅ Authorized as admin. You can approve subscribers with /approve <chat_id> or /approve_all")

async def logout_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lekzy.deauthorize_user(user.id)
    await update.message.reply_text("🔒 Deauthorized (logged out).")

async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    auth = lekzy.is_authorized(user.id)
    await update.message.reply_text(f"User: {user.username or user.first_name}\nID: {user.id}\nAuthorized: {auth}")

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    ok = lekzy.request_subscribe(user.id, user.username or "")
    if ok:
        await update.message.reply_text("✅ Subscription request sent. An admin will review and approve you soon.")
    else:
        await update.message.reply_text("❌ Could not request subscription. Try again later.")

async def list_pending_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not lekzy.is_authorized(user.id):
        await update.message.reply_text("🔒 Admin only command.")
        return
    pending = lekzy.list_pending_subscribers()
    if not pending:
        await update.message.reply_text("No pending subscribers.")
        return
    text = "Pending subscribers:\n" + "\n".join([f"{p['chat_id']} ({p['username']}) requested_at={p['requested_at']}" for p in pending])
    await update.message.reply_text(text)

async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not lekzy.is_authorized(user.id):
        await update.message.reply_text("🔒 Admin only.")
        return
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /approve <chat_id> or /approve_all")
        return
    if args[0] == "all" or args[0] == "ALL" or args[0] == "approve_all":
        pend = lekzy.list_pending_subscribers()
        for p in pend:
            lekzy.approve_subscriber(p['chat_id'])
        await update.message.reply_text(f"✅ Approved {len(pend)} subscribers.")
        return
    try:
        cid = int(args[0])
    except:
        await update.message.reply_text("Invalid chat_id.")
        return
    ok = lekzy.approve_subscriber(cid)
    if ok:
        await update.message.reply_text(f"✅ Approved {cid}.")
    else:
        await update.message.reply_text("❌ Approve failed.")

# settings and reporting
async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    args = context.args
    if not args:
        s = user_settings.get(user.id, {"preentry": PREENTRY_DEFAULT, "timeframe_mode": DEFAULT_TIMEFRAME_MODE})
        await update.message.reply_text(f"Settings:\npreentry={s.get('preentry', PREENTRY_DEFAULT)}s\ntimeframe_mode={s.get('timeframe_mode', DEFAULT_TIMEFRAME_MODE)}")
        return
    try:
        key = args[0].lower()
        if key in ("alert_time","preentry"):
            val = int(args[1])
            user_settings.setdefault(user.id, {"preentry": PREENTRY_DEFAULT, "timeframe_mode": DEFAULT_TIMEFRAME_MODE})
            user_settings[user.id]['preentry'] = max(1, val)
            await update.message.reply_text(f"Pre-entry alert set to {val}s")
            return
        if key == "timeframe":
            val = args[1].lower()
            if val not in ("1m","5m","auto"):
                await update.message.reply_text("timeframe must be '1m','5m' or 'auto'")
                return
            user_settings.setdefault(user.id, {"preentry": PREENTRY_DEFAULT, "timeframe_mode": DEFAULT_TIMEFRAME_MODE})
            user_settings[user.id]['timeframe_mode'] = val
            await update.message.reply_text(f"timeframe_mode set to {val}")
            return
        await update.message.reply_text("Unknown setting. Use /settings or /settings preentry <sec> /settings timeframe <1m|5m|auto>")
    except Exception as e:
        print("settings error:", e)
        await update.message.reply_text("Error updating settings.")

async def report_trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /report_trade <SIGNAL_ID> <EXIT_PRICE>")
        return
    sig = args[0].strip()
    try:
        price = float(args[1])
    except:
        await update.message.reply_text("Invalid exit price.")
        return
    ok = lekzy.record_trade_exit(sig, price)
    if ok:
        await update.message.reply_text("✅ Trade recorded and model updated (if possible).")
    else:
        await update.message.reply_text("❌ Could not record trade; check SIGNAL_ID and that it is OPEN.")

# -------------------- Setup and run --------------------
def setup_app():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(callback_buttons))
    app.add_handler(CommandHandler("login", login_cmd))
    app.add_handler(CommandHandler("logout", logout_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("pending", list_pending_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("report_trade", report_trade_cmd))
    return app

async def main():
    threading.Thread(target=start_http_server, daemon=True).start()
    await lekzy.init_session()
    app = setup_app()
    print("Starting Lekzy_FX_AI_Pro...")
    await app.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
    except Exception as e:
        print("Fatal error:", e)
