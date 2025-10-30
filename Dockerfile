FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./
COPY .env ./

# Create data directory for database
RUN mkdir -p /app/data

# Set execute permissions
RUN chmod +x *.py

# Create a startup script that sets up database and runs the bot
RUN echo '#!/bin/bash\n\
python -c "\n\
import sqlite3\n\
import os\n\
\n\
# Create database directory if it doesnt exist\n\
os.makedirs(\"/app/data\", exist_ok=True)\n\
\n\
# Connect to database\n\
conn = sqlite3.connect(\"/app/data/lekzy_fx_ai.db\")\n\
cursor = conn.cursor()\n\
\n\
# Create subscriptions table\n\
cursor.execute(\"\"\"\n\
    CREATE TABLE IF NOT EXISTS subscriptions (\n\
        user_id INTEGER PRIMARY KEY,\n\
        plan_type TEXT DEFAULT \"TRIAL\",\n\
        start_date TEXT,\n\
        end_date TEXT,\n\
        payment_status TEXT DEFAULT \"PENDING\",\n\
        signals_used INTEGER DEFAULT 0,\n\
        max_daily_signals INTEGER DEFAULT 5,\n\
        allowed_sessions TEXT DEFAULT \"[\\\"MORNING\\\"]\",\n\
        timezone TEXT DEFAULT \"UTC+1\",\n\
        broadcast_enabled INTEGER DEFAULT 1,\n\
        created_at TEXT DEFAULT CURRENT_TIMESTAMP\n\
    )\n\
\"\"\")\n\
\n\
# Create admin_notifications table\n\
cursor.execute(\"\"\"\n\
    CREATE TABLE IF NOT EXISTS admin_notifications (\n\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\n\
        notification_type TEXT,\n\
        user_id INTEGER,\n\
        username TEXT,\n\
        plan_type TEXT,\n\
        details TEXT,\n\
        sent_time TEXT DEFAULT CURRENT_TIMESTAMP\n\
    )\n\
\"\"\")\n\
\n\
# Create user_activity table\n\
cursor.execute(\"\"\"\n\
    CREATE TABLE IF NOT EXISTS user_activity (\n\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\n\
        user_id INTEGER,\n\
        activity_type TEXT,\n\
        details TEXT,\n\
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP\n\
    )\n\
\"\"\")\n\
\n\
conn.commit()\n\
conn.close()\n\
print(\"✅ Database tables created successfully\")\n\
"\n\
\n\
# Start the bot\n\
python lekzy_fx_ai_pro.py\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Run the startup script
CMD ["/bin/bash", "/app/start.sh"]
