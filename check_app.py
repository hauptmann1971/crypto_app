#!/usr/bin/env python3
"""
Скрипт для проверки доступности приложения
Использование: python check_app.py
"""

import sys
import socket
import requests
from pathlib import Path

def check_port(port):
    """Проверяет, открыт ли порт"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def check_app():
    """Проверяет доступность Flask приложения"""
    print("\n🔍 ПРОВЕРКА ДОСТУПНОСТИ ПРИЛОЖЕНИЯ\n")
    print("=" * 60)
    
    # Проверка порта
    print("1️⃣ Проверка порта 5000...")
    if check_port(5000):
        print("   ✅ Порт 5000 ОТКРЫТ (приложение слушает)")
    else:
        print("   ❌ Порт 5000 ЗАКРЫТ (приложение не запущено)")
        return False
    
    # Проверка HTTP запросом
    print("\n2️⃣ Проверка HTTP запроса...")
    try:
        response = requests.get('http://localhost:5000/', timeout=3)
        print(f"   ✅ HTTP Статус: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Приложение вернуло успешный ответ")
            
            # Проверка Content-Type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print(f"   ✅ Content-Type: {content_type} (HTML страница)")
            
            # Краткое содержимое
            content_preview = response.text[:200].replace('\n', ' ').strip()
            print(f"   📄 Начало страницы: {content_preview}...")
            
        elif response.status_code == 302:
            print(f"   ⚠️  Редирект (302) на: {response.headers.get('Location', 'unknown')}")
            
        elif response.status_code >= 400:
            print(f"   ⚠️  Ошибка сервера: {response.status_code}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Ошибка подключения: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Таймаут запроса (>3 сек)")
        return False
    except Exception as e:
        print(f"   ❌ Неожиданная ошибка: {e}")
        return False
    
    # Проверка статуса системы
    print("\n3️⃣ Проверка endpoint /status...")
    try:
        status_response = requests.get('http://localhost:5000/status', timeout=3)
        if status_response.status_code == 200:
            print("   ✅ /status доступен")
            import json
            try:
                status_data = status_response.json()
                print(f"   📊 DB подключен: {status_data.get('db_connected', 'N/A')}")
                print(f"   📊 Worker активен: {status_data.get('worker_active', 'N/A')}")
            except:
                pass
        else:
            print(f"   ⚠️  /status вернул {status_response.status_code}")
    except Exception as e:
        print(f"   ⚠️  /status недоступен: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ВЫВОД: Приложение ДОСТУПНО из браузера!")
    print("\n🌐 Откройте в браузере: http://localhost:5000/")
    print("=" * 60 + "\n")
    
    return True

if __name__ == '__main__':
    try:
        success = check_app()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Проверка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка при проверке: {e}")
        sys.exit(1)
