import os
import json
import pytz

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8322565044:AAFlcvasvVumqjYMGbngp7rfW3_ScANrCfM")
    ADMIN_USER_IDS = json.loads(os.getenv("ADMIN_USER_IDS", "[6307001401,6957180168]"))
    DB_PATH = "/app/data/lekzy_fx_ai.db"  # Docker path
    TZ = pytz.timezone('Europe/London')
    MIN_COOLDOWN = 180
    MAX_COOLDOWN = 600
    MIN_CONFIDENCE = 0.85
    
    NOTIFY_NEW_SUBSCRIBERS = os.getenv("NOTIFY_NEW_SUBSCRIBERS", "true").lower() == "true"
    NOTIFY_UPGRADES = os.getenv("NOTIFY_UPGRADES", "true").lower() == "true"
    NOTIFY_TRIAL_EXPIRY = os.getenv("NOTIFY_TRIAL_EXPIRY", "true").lower() == "true"
    NOTIFY_PAYMENT_ISSUES = os.getenv("NOTIFY_PAYMENT_ISSUES", "true").lower() == "true"
    NOTIFY_SYSTEM_ALERTS = os.getenv("NOTIFY_SYSTEM_ALERTS", "true").lower() == "true"
