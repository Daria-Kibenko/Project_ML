import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecopulse.config import cfg
from telethon.sync import TelegramClient

print("=== Авторизация в Telegram ===")
print(f"API_ID:   {cfg.API_ID}")
print(f"Сессия:   {cfg.SESSION_NAME}.session")
print()

with TelegramClient(cfg.SESSION_NAME, cfg.API_ID, cfg.API_HASH) as client:
    me = client.get_me()
    print(f"✅ Авторизация успешна!")
    print(f"   Аккаунт: {me.first_name} {me.last_name or ''}")
    print(f"   Username: @{me.username or 'нет'}")
    print(f"   Телефон: {me.phone}")
    print()
    print(f"Файл сессии сохранён: {cfg.SESSION_NAME}.session")
    print("Теперь можно запускать: python main.py")
