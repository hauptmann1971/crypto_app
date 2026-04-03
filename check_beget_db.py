#!/usr/bin/env python3
"""Проверка подключения к MySQL на Beget"""

import pymysql
import socket

# Твои данные из .env
DB_HOST = "kalmyk3j.beget.tech"
DB_USER = "kalmyk3j_romanov"
DB_PASSWORD = "6xkGG33NX9%p"
DB_NAME = "kalmyk3j_romanov"
DB_PORT = 3306

print("\n" + "=" * 60)
print("📡 ПРОВЕРКА ПОДКЛЮЧЕНИЯ К MYSQL НА BEGET")
print("=" * 60)

# 1. DNS резолвинг
print("\n1️⃣ DNS резолвинг:")
try:
    ip = socket.gethostbyname(DB_HOST)
    print(f"   ✅ {DB_HOST} → {ip}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 2. TCP подключение
print("\n2️⃣ TCP подключение к порту 3306:")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    result = sock.connect_ex((DB_HOST, DB_PORT))
    sock.close()
    
    if result == 0:
        print(f"   ✅ Порт {DB_PORT} открыт")
    else:
        print(f"   ❌ Порт {DB_PORT} недоступен (код: {result})")
        print("   → На Beget внешний доступ к MySQL может быть ограничен")
        print("   → Используй 'kalmyk3j.beget.tech' или IP из панели")
        exit(1)
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 3. MySQL авторизация
print("\n3️⃣ MySQL авторизация:")
try:
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        connect_timeout=10,
        charset='utf8mb4'
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION(), DATABASE(), USER(), NOW()")
    version, db_name, user, now = cursor.fetchone()
    
    print(f"   ✅ ПОДКЛЮЧЕНИЕ УСПЕШНО!")
    print(f"   Версия MySQL: {version}")
    print(f"   База данных: {db_name}")
    print(f"   Пользователь: {user}")
    print(f"   Серверное время: {now}")
    
    # Проверим таблицы
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print(f"   Таблиц в БД: {len(tables)}")
    for table in tables[:5]:
        print(f"      - {table[0]}")
    
    cursor.close()
    conn.close()
    
except pymysql.err.OperationalError as e:
    code = e.args[0]
    print(f"   ❌ Ошибка: {e}")
    
    if code == 1045:
        print("   → Неверный пользователь или пароль")
        print("   → Проверь логин и пароль в панели Beget")
    elif code == 1049:
        print("   → База данных не найдена")
    elif code == 2003:
        print("   → Не удалось подключиться к серверу")
        print("   → Beget может требовать подключение через SSL")
        print("   → Попробуй: kalmyk3j.beget.tech или IP из панели")
    elif code == 2013:
        print("   → Потеря соединения — возможно, нужно добавить ?connect_timeout=30")
    else:
        print(f"   → Код ошибки: {code}")
    
    exit(1)
except Exception as e:
    print(f"   ❌ Другая ошибка: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ БД ДОСТУПНА! Можешь запускать приложение.")
print("=" * 60)