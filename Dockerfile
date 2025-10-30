FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./
COPY .env ./

# Create necessary directories and set permissions
RUN mkdir -p /app/data && \
    chmod +x *.py

# Create and set up database on startup
RUN python -c "
import sqlite3
conn = sqlite3.connect('/app/data/lekzy_fx_ai.db')
cursor = conn.cursor()

# Create essential tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS subscriptions (
        user_id INTEGER PRIMARY KEY,
        plan_type TEXT DEFAULT 'TRIAL',
        start_date TEXT,
        end_date TEXT,
        payment_status TEXT DEFAULT 'PENDING',
        signals_used INTEGER DEFAULT 0,
        max_daily_signals INTEGER DEFAULT 5,
        allowed_sessions TEXT DEFAULT '[\"MORNING\"]',
        timezone TEXT DEFAULT 'UTC+1',
        broadcast_enabled INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin_notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        notification_type TEXT,
        user_id INTEGER,
        username TEXT,
        plan_type TEXT,
        details TEXT,
        sent_time TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        activity_type TEXT,
        details TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.commit()
conn.close()
print('✅ Database tables created successfully')
"

# Run the bot
CMD ["python", "lekzy_fx_ai_pro.py"]
