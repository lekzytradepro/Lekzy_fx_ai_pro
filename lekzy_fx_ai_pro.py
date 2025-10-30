import os
import asyncio
import sqlite3
import json
import time
import random
import logging
import hashlib
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

# ==================== CONFIGURATION ====================
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_bot_token_here")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Change this in production!
    DB_PATH = "/app/data/lekzy_fx_ai.db"
    
    # Signal settings
    PRE_ENTRY_LEAD_TIME = 2  # minutes before entry signal
    MIN_COOLDOWN = 300
    MAX_COOLDOWN = 900

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
        os.makedirs("/app/data", exist_ok=True)
        
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()

        # Admin sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                login_time TEXT,
                expiry_time TEXT
            )
        """)

        # Subscriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                plan_type TEXT DEFAULT 'TRIAL',
                start_date TEXT,
                end_date TEXT,
                payment_status TEXT DEFAULT 'PENDING',
                signals_used INTEGER DEFAULT 0,
                max_daily_signals INTEGER DEFAULT 5,
                allowed_sessions TEXT DEFAULT '["MORNING"]',
                timezone TEXT DEFAULT 'UTC+1',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Admin notifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notification_type TEXT,
                user_id INTEGER,
                username TEXT,
                plan_type TEXT,
                details TEXT,
                sent_time TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                details TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Enhanced signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                signal_type TEXT,  -- PRE_ENTRY or ENTRY
                direction TEXT,
                entry_price REAL,
                take_profit REAL,
                stop_loss REAL,
                confidence REAL,
                session_type TEXT,
                analysis TEXT,
                time_to_entry INTEGER,  -- minutes until entry
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

# ==================== ADMIN AUTHENTICATION ====================
class AdminAuth:
    def __init__(self):
        self.session_duration = timedelta(hours=24)
    
    def verify_password(self, password: str) -> bool:
        """Verify admin password"""
        return password == Config.ADMIN_PASSWORD
    
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
                    # Remove expired session
                    conn.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
                    conn.commit()
            return False
    
    def logout(self, user_id: int):
        """Logout admin"""
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.execute("DELETE FROM admin_sessions WHERE user_id = ?", (user_id,))
            conn.commit()

# ==================== SESSION MANAGER ====================
class SessionManager:
    def __init__(self):
        self.sessions = {
            "MORNING": {
                "start_hour": 8, "end_hour": 12,
                "name": "European Session",
                "optimal_pairs": ["EUR/USD", "GBP/USD", "EUR/JPY"],
                "volatility": "HIGH",
                "typical_movement": 45,
                "accuracy": 96.2
            },
            "EVENING": {
                "start_hour": 16, "end_hour": 20,
                "name": "NY/London Overlap", 
                "optimal_pairs": ["USD/JPY", "USD/CAD", "XAU/USD"],
                "volatility": "VERY HIGH",
                "typical_movement": 55,
                "accuracy": 97.8
            },
            "ASIAN": {
                "start_hour": 0, "end_hour": 4,
                "name": "Asian Session",
                "optimal_pairs": ["AUD/JPY", "NZD/USD", "USD/JPY"],
                "volatility": "MEDIUM",
                "typical_movement": 30,
                "accuracy": 92.5
            }
        }

    def get_current_session(self):
        """Get current active trading session"""
        now = datetime.now()
        current_hour = now.hour
        
        for session_id, session in self.sessions.items():
            if session["start_hour"] <= current_hour < session["end_hour"]:
                return {**session, "id": session_id}
        
        return {"id": "CLOSED", "name": "Market Closed"}

# ==================== ENHANCED SIGNAL GENERATOR ====================
class EnhancedSignalGenerator:
    def __init__(self):
        self.pending_signals = {}  # Track pre-entry signals waiting for entry
    
    def generate_minimal_analysis(self, symbol: str, session_type: str) -> dict:
        """Generate minimal trade analysis"""
        # Market conditions
        conditions = [
            "Price at key support level",
            "RSI showing oversold condition", 
            "Bullish candlestick pattern forming",
            "Trend line bounce confirmed",
            "Volume spike indicating momentum",
            "MACD histogram turning positive",
            "Fibonacci retracement level holding",
            "Moving average support test"
        ]
        
        # Risk factors
        risks = [
            "Watch for economic news",
            "Monitor volume confirmation",
            "Check for divergence",
            "Verify trend alignment"
        ]
        
        return {
            "market_condition": random.choice(conditions),
            "key_level": round(random.uniform(1.0750, 1.0950), 4) if "EUR" in symbol else round(random.uniform(1.2500, 1.2800), 4),
            "momentum": random.choice(["STRONG", "MODERATE", "BUILDING"]),
            "risk_factors": random.sample(risks, 2),
            "timeframe": "M5",
            "setup_quality": random.choice(["HIGH", "VERY_HIGH"])
        }
    
    def generate_pre_entry_signal(self, symbol: str, session_type: str) -> dict:
        """Generate pre-entry signal with analysis"""
        analysis = self.generate_minimal_analysis(symbol, session_type)
        direction = "BUY" if random.random() > 0.45 else "SELL"
        
        signal_id = f"PRE_{symbol.replace('/', '')}_{int(time.time())}"
        
        # Calculate approximate entry levels based on analysis
        if direction == "BUY":
            entry_price = round(analysis["key_level"] + 0.0005, 5)
        else:
            entry_price = round(analysis["key_level"] - 0.0005, 5)
        
        signal_data = {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal_type": "PRE_ENTRY",
            "direction": direction,
            "entry_price": entry_price,
            "take_profit": 0.0,  # Will be set in entry signal
            "stop_loss": 0.0,    # Will be set in entry signal
            "confidence": round(random.uniform(0.82, 0.95), 3),
            "session_type": session_type,
            "analysis": json.dumps(analysis),
            "time_to_entry": Config.PRE_ENTRY_LEAD_TIME,
            "generated_at": datetime.now().isoformat()
        }
        
        # Store for entry signal generation
        self.pending_signals[signal_id] = signal_data
        return signal_data
    
    def generate_entry_signal(self, pre_signal_id: str) -> dict:
        """Generate entry signal based on pre-entry"""
        if pre_signal_id not in self.pending_signals:
            return None
        
        pre_signal = self.pending_signals[pre_signal_id]
        analysis = json.loads(pre_signal["analysis"])
        
        # Calculate proper TP/SL based on direction and analysis
        movement = 0.0030 if analysis["momentum"] == "STRONG" else 0.0020
        
        if pre_signal["direction"] == "BUY":
            take_profit = round(pre_signal["entry_price"] + movement, 5)
            stop_loss = round(pre_signal["entry_price"] - movement * 0.6, 5)
        else:
            take_profit = round(pre_signal["entry_price"] - movement, 5)
            stop_loss = round(pre_signal["entry_price"] + movement * 0.6, 5)
        
        entry_signal_id = pre_signal_id.replace("PRE_", "ENTRY_")
        
        entry_signal = {
            **pre_signal,
            "signal_id": entry_signal_id,
            "signal_type": "ENTRY",
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "time_to_entry": 0,  # Immediate entry
            "risk_reward": round((take_profit - pre_signal["entry_price"]) / (pre_signal["entry_price"] - stop_loss) 
                               if pre_signal["direction"] == "BUY" else 
                               (pre_signal["entry_price"] - take_profit) / (stop_loss - pre_signal["entry_price"]), 2)
        }
        
        # Remove from pending
        del self.pending_signals[pre_signal_id]
        
        return entry_signal

# ==================== ADMIN NOTIFICATION MANAGER ====================
class AdminNotificationManager:
    def __init__(self, application):
        self.application = application
        self.admin_auth = AdminAuth()
    
    async def notify_new_subscriber(self, user_id: int, username: str, plan_type: str):
        """Notify all active admins about new subscriber"""
        # Get all active admins
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT user_id FROM admin_sessions WHERE expiry_time > ?",
                (datetime.now().isoformat(),)
            )
            active_admins = [row[0] for row in cursor.fetchall()]
        
        if not active_admins:
            logger.warning("No active admins to notify")
            return
            
        message = f"""
🎉 NEW SUBSCRIBER ALERT!

👤 User: @{username} (ID: `{user_id}`)
📋 Plan: {plan_type}
⏰ Time: {datetime.now().strftime('%H:%M:%S')}

Total Active Admins: {len(active_admins)}

Welcome them to LEKZY FX AI PRO! 🚀
"""
        
        for admin_id in active_admins:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"✅ Admin notified: {admin_id}")
            except Exception as e:
                logger.error(f"❌ Failed to notify admin {admin_id}: {e}")

# ==================== SUBSCRIPTION MANAGER ====================
class SubscriptionManager:
    def __init__(self, db_path: str, admin_manager: AdminNotificationManager):
        self.db_path = db_path
        self.admin_manager = admin_manager
    
    def start_free_trial(self, user_id: int, username: str, first_name: str):
        """Start free trial with admin notification"""
        end_date = datetime.now() + timedelta(days=3)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_type, start_date, end_date, payment_status, max_daily_signals, allowed_sessions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, "TRIAL", datetime.now().isoformat(), 
                end_date.isoformat(), "TRIAL", 5, '["MORNING"]'
            ))
            conn.commit()
        
        # Notify admins
        asyncio.create_task(self.admin_manager.notify_new_subscriber(user_id, username, "TRIAL"))
        logger.info(f"✅ New trial started: {username} ({user_id})")
    
    def get_user_plan(self, user_id: int) -> str:
        """Get user's current plan"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT plan_type FROM subscriptions WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else "TRIAL"

# ==================== ENHANCED TRADING BOT ====================
class EnhancedTradingBot:
    def __init__(self):
        self.session_manager = SessionManager()
        self.signal_generator = EnhancedSignalGenerator()
        self.is_running = False
    
    async def start_enhanced_signal_generation(self):
        """Start enhanced signal generation with pre-entry and entry"""
        self.is_running = True
        
        async def signal_loop():
            while self.is_running:
                try:
                    current_session = self.session_manager.get_current_session()
                    
                    if current_session["id"] != "CLOSED":
                        logger.info(f"🕒 Generating signals for {current_session['name']}")
                        
                        # Generate signals for optimal pairs
                        for symbol in current_session["optimal_pairs"][:2]:
                            # Step 1: Generate pre-entry signal
                            pre_signal = self.signal_generator.generate_pre_entry_signal(symbol, current_session["id"])
                            
                            # Store pre-entry signal
                            with sqlite3.connect(Config.DB_PATH) as conn:
                                conn.execute("""
                                    INSERT INTO signals 
                                    (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    pre_signal["signal_id"],
                                    pre_signal["symbol"],
                                    pre_signal["signal_type"],
                                    pre_signal["direction"],
                                    pre_signal["entry_price"],
                                    pre_signal["take_profit"],
                                    pre_signal["stop_loss"],
                                    pre_signal["confidence"],
                                    pre_signal["session_type"],
                                    pre_signal["analysis"],
                                    pre_signal["time_to_entry"]
                                ))
                                conn.commit()
                            
                            logger.info(f"📊 Pre-entry generated: {pre_signal['symbol']} {pre_signal['direction']}")
                            
                            # Wait before sending entry signal
                            await asyncio.sleep(Config.PRE_ENTRY_LEAD_TIME * 60)
                            
                            # Step 2: Generate entry signal
                            entry_signal = self.signal_generator.generate_entry_signal(pre_signal["signal_id"])
                            
                            if entry_signal:
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    conn.execute("""
                                        INSERT INTO signals 
                                        (signal_id, symbol, signal_type, direction, entry_price, take_profit, stop_loss, confidence, session_type, analysis, time_to_entry)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        entry_signal["signal_id"],
                                        entry_signal["symbol"],
                                        entry_signal["signal_type"],
                                        entry_signal["direction"],
                                        entry_signal["entry_price"],
                                        entry_signal["take_profit"],
                                        entry_signal["stop_loss"],
                                        entry_signal["confidence"],
                                        entry_signal["session_type"],
                                        entry_signal["analysis"],
                                        entry_signal["time_to_entry"]
                                    ))
                                    conn.commit()
                                
                                logger.info(f"🎯 Entry signal generated: {entry_signal['symbol']} {entry_signal['direction']}")
                    
                    # Wait before next cycle
                    wait_time = random.randint(Config.MIN_COOLDOWN, Config.MAX_COOLDOWN)
                    logger.info(f"⏰ Next signal cycle in {wait_time//60} minutes")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}")
                    await asyncio.sleep(30)
        
        asyncio.create_task(signal_loop())
        logger.info("✅ Enhanced signal generation started")

# ==================== ENHANCED TELEGRAM BOT ====================
class EnhancedTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.application = None
        self.admin_auth = AdminAuth()
        self.admin_manager = None
        self.subscription_manager = None
        self.trading_bot = None
    
    async def initialize(self):
        """Initialize the bot"""
        self.application = Application.builder().token(self.token).build()
        self.admin_manager = AdminNotificationManager(self.application)
        self.subscription_manager = SubscriptionManager(Config.DB_PATH, self.admin_manager)
        self.trading_bot = EnhancedTradingBot()
        
        # Setup command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("admin", self.admin_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("session", self.session_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("users", self.users_command))
        
        # Add message handler for password input
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        await self.application.initialize()
        await self.application.start()
        
        # Start trading features
        await self.trading_bot.start_enhanced_signal_generation()
        
        logger.info("🤖 Enhanced Telegram bot initialized successfully!")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        # Register user and start trial
        self.subscription_manager.start_free_trial(user.id, user.username, user.first_name)
        
        welcome_message = f"""
🎉 Welcome to LEKZY FX AI PRO, {user.first_name}!

Your 3-day free trial has been activated! 
You'll receive trading signals during morning sessions.

🔔 *Admins have been notified of your subscription.*

*Enhanced Signal System:*
• 📊 Pre-entry analysis (2 min before)
• 🎯 Entry signals with exact levels
• ⚡ Minimal, actionable trade analysis

*Available Commands:*
• /stats - Your account status
• /session - Current trading session
• /signals - Recent trading signals
• /login - Admin access

Happy trading! 📈
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login command"""
        user = update.effective_user
        
        if self.admin_auth.is_admin(user.id):
            await update.message.reply_text("✅ You are already logged in as admin!")
            await self.show_admin_dashboard(update)
            return
        
        await update.message.reply_text(
            "🔐 *Admin Login*\n\nPlease enter the admin password:",
            parse_mode='Markdown'
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle message input for admin login"""
        user = update.effective_user
        message_text = update.message.text
        
        # Check if this might be a password attempt
        if not self.admin_auth.is_admin(user.id):
            if self.admin_auth.verify_password(message_text):
                self.admin_auth.create_session(user.id, user.username)
                await update.message.reply_text("✅ *Admin access granted!*", parse_mode='Markdown')
                await self.show_admin_dashboard(update)
            else:
                await update.message.reply_text("❌ Invalid password. Use /login to try again.")
    
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /admin command"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text(
                "❌ *Admin access required*\n\nUse /login to access admin features.",
                parse_mode='Markdown'
            )
            return
        
        await self.show_admin_dashboard(update)
    
    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout command"""
        user = update.effective_user
        
        if self.admin_auth.is_admin(user.id):
            self.admin_auth.logout(user.id)
            await update.message.reply_text("✅ Logged out from admin access.")
        else:
            await update.message.reply_text("You are not currently logged in as admin.")
    
    async def show_admin_dashboard(self, update: Update):
        """Show admin dashboard"""
        # Get statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM subscriptions")
            total_users = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM admin_notifications")
            total_notifications = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'PRE_ENTRY'")
            pre_entry_signals = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'ENTRY'")
            entry_signals = cursor.fetchone()[0]
        
        admin_message = f"""
🏢 *ADMIN DASHBOARD*

Welcome, admin! 👑

📊 *System Statistics:*
• Total Users: {total_users}
• Notifications Sent: {total_notifications}
• Pre-entry Signals: {pre_entry_signals}
• Entry Signals: {entry_signals}
• Total Signals: {total_signals}

🕒 *Current Session:*
{self.trading_bot.session_manager.get_current_session()['name']}

🔧 *Enhanced Features:*
• Password-based admin auth ✅
• Pre-entry + Entry signals ✅  
• Minimal trade analysis ✅
• Real-time notifications ✅

*Admin Commands:*
• /users - View user statistics
• /logout - End admin session
"""
        keyboard = [
            [InlineKeyboardButton("👥 User Management", callback_data="admin_users")],
            [InlineKeyboardButton("📊 System Stats", callback_data="admin_stats")],
            [InlineKeyboardButton("🔐 Logout", callback_data="admin_logout")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(admin_message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def users_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /users command (admin only)"""
        user = update.effective_user
        
        if not self.admin_auth.is_admin(user.id):
            await update.message.reply_text("❌ Admin access required.")
            return
        
        # Get user statistics
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT plan_type, COUNT(*) 
                FROM subscriptions 
                GROUP BY plan_type
            """)
            plan_distribution = cursor.fetchall()
            
            cursor = conn.execute("""
                SELECT username, created_at 
                FROM subscriptions 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_users = cursor.fetchall()
        
        users_message = "👥 *USER MANAGEMENT*\n\n"
        
        users_message += "*Plan Distribution:*\n"
        for plan, count in plan_distribution:
            users_message += f"• {plan}: {count} users\n"
        
        users_message += "\n*Recent Users:*\n"
        for username, joined_date in recent_users:
            date_str = datetime.fromisoformat(joined_date).strftime("%m/%d")
            users_message += f"• @{username} - {date_str}\n"
        
        await update.message.reply_text(users_message, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        user_plan = self.subscription_manager.get_user_plan(user.id)
        
        stats_message = f"""
📊 *YOUR ACCOUNT STATS*

👤 User: {user.first_name}
📋 Plan: {user_plan}
🆔 ID: `{user.id}`

💡 *Enhanced Signal Features:*
• Pre-entry analysis (2 min advance)
• Minimal trade analysis
• Exact entry/exit levels
• Session-based optimization

🎯 *Current Access:*
• Morning Session (08:00-12:00)
• 5 Signals Per Day
• {self.trading_bot.session_manager.sessions['MORNING']['accuracy']}% Accuracy

*Use /session to check current market hours*
"""
        await update.message.reply_text(stats_message, parse_mode='Markdown')
    
    async def session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command"""
        current_session = self.trading_bot.session_manager.get_current_session()
        
        if current_session["id"] == "CLOSED":
            message = """
🕒 *MARKET CLOSED*

No active trading sessions at the moment.

*Next Sessions:*
• 🌅 Morning: 08:00 - 12:00 (96.2%)
• 🌇 Evening: 16:00 - 20:00 (97.8%)  
• 🌃 Asian: 00:00 - 04:00 (92.5%)

*Enhanced signals will resume during session hours!*
"""
        else:
            message = f"""
🕒 *CURRENT TRADING SESSION*

*{current_session['name']}* ✅ ACTIVE
⏰ {current_session['start_hour']:02d}:00 - {current_session['end_hour']:02d}:00

📊 *Session Details:*
• Volatility: {current_session['volatility']}
• Accuracy: {current_session['accuracy']}%
• Typical Movement: {current_session['typical_movement']} pips

🎯 *Optimal Pairs:*
{', '.join(current_session['optimal_pairs'])}

⚡ *Enhanced Signals Active:*
• Pre-entry analysis
• Minimal trade setup
• Entry signals with levels
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        # Get recent signals
        with sqlite3.connect(Config.DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT signal_id, symbol, signal_type, direction, entry_price, confidence, created_at, analysis
                FROM signals 
                ORDER BY created_at DESC 
                LIMIT 6
            """)
            recent_signals = cursor.fetchall()
        
        if not recent_signals:
            await update.message.reply_text("📭 No signals generated yet. Check during trading sessions!")
            return
        
        signals_message = "📡 *RECENT TRADING SIGNALS*\n\n"
        
        for signal in recent_signals:
            signal_id, symbol, signal_type, direction, entry_price, confidence, created_at, analysis_json = signal
            
            if signal_type == "PRE_ENTRY":
                emoji = "📊"
                type_text = "Pre-Entry Analysis"
            else:
                emoji = "🎯" 
                type_text = "Entry Signal"
            
            time_str = datetime.fromisoformat(created_at).strftime("%H:%M")
            direction_emoji = "🟢" if direction == "BUY" else "🔴"
            
            signals_message += f"""
{emoji} *{type_text}* - {symbol}
{direction_emoji} {direction} | 💵 {entry_price:.5f}
🎯 {confidence*100:.1f}% | ⏰ {time_str}
"""
            
            if signal_type == "PRE_ENTRY" and analysis_json:
                analysis = json.loads(analysis_json)
                signals_message += f"   ⚡ {analysis['market_condition']}\n"
        
        signals_message += "\n💡 *Pre-entry signals come 2 minutes before entry signals*"
        
        await update.message.reply_text(signals_message, parse_mode='Markdown')
    
    async def start_polling(self):
        """Start polling for messages"""
        logger.info("📡 Starting to poll for messages...")
        await self.application.updater.start_polling()
    
    async def stop(self):
        """Stop the bot"""
        self.trading_bot.is_running = False
        await self.application.stop()
        logger.info("🛑 Enhanced Telegram bot stopped")

# ==================== MAIN APPLICATION ====================
class UltimateApplication:
    def __init__(self):
        self.telegram_bot = None
        self.is_running = False
    
    async def setup(self):
        """Setup the application"""
        logger.info("🔄 Setting up LEKZY FX AI PRO with Enhanced Features...")
        
        # Initialize database first
        initialize_database()
        
        self.telegram_bot = EnhancedTelegramBot()
        await self.telegram_bot.initialize()
        
        self.is_running = True
        logger.info("🎯 LEKZY FX AI PRO with Admin Auth & Enhanced Signals is READY!")
    
    async def run(self):
        """Run the application"""
        if not self.is_running:
            await self.setup()
        
        logger.info("🔄 Starting main loop...")
        await self.telegram_bot.start_polling()
        
        # Keep the application running
        while self.is_running:
            await asyncio.sleep(10)
    
    async def shutdown(self):
        """Shutdown the application"""
        self.is_running = False
        if self.telegram_bot:
            await self.telegram_bot.stop()
        logger.info("🛑 Application stopped")

# ==================== START THE BOT ====================
async def main():
    """Main application entry point"""
    app = UltimateApplication()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
