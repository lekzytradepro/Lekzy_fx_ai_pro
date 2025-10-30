import sqlite3
import os

def setup_database():
    """Simple database setup for Render"""
    conn = sqlite3.connect('lekzy_fx_ai.db')
    cursor = conn.cursor()
    
    # Essential tables only
    tables = [
        # Subscriptions table
        """
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
            broadcast_enabled INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Admin notifications table
        """
        CREATE TABLE IF NOT EXISTS admin_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_type TEXT,
            user_id INTEGER,
            username TEXT,
            plan_type TEXT,
            details TEXT,
            sent_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    for table_sql in tables:
        cursor.execute(table_sql)
    
    conn.commit()
    conn.close()
    print("✅ Database setup completed")

if __name__ == "__main__":
    setup_database()
