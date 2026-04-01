#!/usr/bin/env python3
"""
Скрипт для управления процессами приложения и воркера
Использование:
    python process_manager.py start     - Запустить оба процесса
    python process_manager.py stop      - Остановить оба процесса
    python process_manager.py restart   - Перезапустить оба процесса
    python process_manager.py status    - Проверить статус
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()
PID_DIR = PROJECT_ROOT / ".pids"
APP_PID_FILE = PID_DIR / "app.pid"
WORKER_PID_FILE = PID_DIR / "worker.pid"

# Константы
APP_SCRIPT = "app.py"
WORKER_SCRIPT = "worker.py"
RESTART_DELAY_APP = 1  # секунды между остановкой и запуском
RESTART_DELAY_WORKER = 2  # секунды между запусками

def ensure_pid_dir():
    """Создаёт директорию для PID файлов если её нет"""
    try:
        PID_DIR.mkdir(exist_ok=True)
    except PermissionError as e:
        logger.error(f"❌ Ошибка создания директории .pids: {e}")
        return False
    return True

def validate_script(script_name):
    """Проверяет существование скрипта"""
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        logger.error(f"❌ Скрипт не найден: {script_path}")
        return False
    return True

def get_process_by_pid(pid_file):
    """Получает процесс по PID файлу"""
    if not pid_file.exists():
        return None
    
    try:
        content = pid_file.read_text().strip()
        if not content.isdigit():
            logger.warning(f"⚠️  Повреждён PID файл {pid_file}, удаляю")
            pid_file.unlink()
            return None
            
        pid = int(content)
        
        # Проверяем, существует ли процесс
        os.kill(pid, 0)
        
        # Дополнительная проверка - читаем имя процесса
        if sys.platform == 'win32':
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if str(pid) not in result.stdout:
                    logger.warning(f"⚠️  Процесс {pid} не найден в списке задач")
                    pid_file.unlink()
                    return None
            except Exception as e:
                logger.debug(f"Не удалось проверить процесс через tasklist: {e}")
        
        return pid
        
    except ProcessLookupError:
        # Процесс не существует
        logger.debug(f"Процесс {pid_file.name} не найден, удаляю PID файл")
        if pid_file.exists():
            pid_file.unlink()
        return None
    except ValueError as e:
        logger.error(f"❌ Ошибка чтения PID из {pid_file}: {e}")
        if pid_file.exists():
            pid_file.unlink()
        return None
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при чтении {pid_file}: {e}")
        return None

def kill_process(pid, process_name):
    """Убивает процесс по PID с проверкой результата"""
    if not pid:
        logger.warning(f"⚠️  Попытка остановки процесса с PID=None")
        return False
    
    try:
        if sys.platform == 'win32':
            # Windows - используем taskkill и проверяем результат
            result = subprocess.run(
                ['taskkill', '/F', '/PID', str(pid)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                if "ERROR:" in result.stderr or "не удается найти" in result.stderr.lower():
                    logger.info(f"ℹ️  Процесс {process_name} (PID: {pid}) уже остановлен")
                    return True  # Считаем успешным, т.к. процесса нет
                else:
                    logger.error(f"❌ Ошибка остановки {process_name}: {result.stderr}")
                    return False
            else:
                logger.info(f"✅ {process_name} (PID: {pid}) успешно остановлен")
                return True
        else:
            # Linux/Mac
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info(f"✅ {process_name} (PID: {pid}) остановлен")
            return True
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Таймаут при остановке {process_name} (PID: {pid})")
        return False
    except ProcessLookupError:
        logger.info(f"ℹ️  Процесс {process_name} (PID: {pid}) уже не существует")
        return True
    except PermissionError:
        logger.error(f"❌ Нет прав для остановки процесса {process_name} (PID: {pid})")
        return False
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при остановке {process_name}: {e}")
        return False

def start_app():
    """Запускает Flask приложение с обработкой ошибок"""
    # Проверяем, запущен ли уже процесс
    pid = get_process_by_pid(APP_PID_FILE)
    if pid:
        logger.info(f"ℹ️  Приложение уже запущено (PID: {pid})")
        return True
    
    # Проверяем наличие скрипта
    if not validate_script(APP_SCRIPT):
        return False
    
    logger.info("🚀 Запуск Flask приложения...")
    
    try:
        if sys.platform == 'win32':
            proc = subprocess.Popen(
                [sys.executable, APP_SCRIPT],
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, APP_SCRIPT],
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Ждём немного и проверяем, не упал ли процесс сразу
        time.sleep(0.5)
        try:
            os.kill(proc.pid, 0)
            # Процесс жив
            APP_PID_FILE.write_text(str(proc.pid))
            logger.info(f"✅ Приложение запущено (PID: {proc.pid})")
            return True
        except ProcessLookupError:
            logger.error("❌ Процесс приложения завершился сразу после запуска")
            return False
            
    except FileNotFoundError as e:
        logger.error(f"❌ Python не найден: {e}")
        return False
    except PermissionError as e:
        logger.error(f"❌ Нет прав для запуска: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка запуска приложения: {e}")
        return False

def start_worker():
    """Запускает Worker с обработкой ошибок"""
    pid = get_process_by_pid(WORKER_PID_FILE)
    if pid:
        logger.info(f"ℹ️  Воркер уже запущен (PID: {pid})")
        return True
    
    if not validate_script(WORKER_SCRIPT):
        return False
    
    logger.info("⚙️  Запуск Worker'а...")
    
    try:
        if sys.platform == 'win32':
            proc = subprocess.Popen(
                [sys.executable, WORKER_SCRIPT],
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, WORKER_SCRIPT],
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Проверяем, не упал ли процесс сразу
        time.sleep(0.5)
        try:
            os.kill(proc.pid, 0)
            WORKER_PID_FILE.write_text(str(proc.pid))
            logger.info(f"✅ Worker запущен (PID: {proc.pid})")
            return True
        except ProcessLookupError:
            logger.error("❌ Процесс воркера завершился сразу после запуска")
            return False
            
    except FileNotFoundError as e:
        logger.error(f"❌ Python не найден: {e}")
        return False
    except PermissionError as e:
        logger.error(f"❌ Нет прав для запуска: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка запуска воркера: {e}")
        return False

def stop_app():
    """Останавливает Flask приложение"""
    pid = get_process_by_pid(APP_PID_FILE)
    if not pid:
        logger.info("ℹ️  Приложение не запущено")
        return True  # Считаем успешным
    
    success = kill_process(pid, "Приложение")
    if success and APP_PID_FILE.exists():
        APP_PID_FILE.unlink()
    return success

def stop_worker():
    """Останавливает Worker"""
    pid = get_process_by_pid(WORKER_PID_FILE)
    if not pid:
        logger.info("ℹ️  Воркер не запущен")
        return True
    
    success = kill_process(pid, "Worker")
    if success and WORKER_PID_FILE.exists():
        WORKER_PID_FILE.unlink()
    return success

def show_status():
    """Показывает статус процессов с детальной информацией"""
    print("\n📊 СТАТУС ПРОЦЕССОВ:")
    print("-" * 60)
    
    app_pid = get_process_by_pid(APP_PID_FILE)
    worker_pid = get_process_by_pid(WORKER_PID_FILE)
    
    app_running = False
    worker_running = False
    
    if app_pid:
        print(f"✅ Приложение:     PID {app_pid}")
        app_running = True
        
        # Дополнительная информация для Windows
        if sys.platform == 'win32':
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {app_pid}', '/FO', 'CSV'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if app_pid in result.stdout:
                    # Парсим CSV вывод
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 4:
                        mem_usage = parts[4].replace('"', '').strip()
                        print(f"                  Память: {mem_usage}")
            except Exception:
                pass
    else:
        print(f"❌ Приложение:     Не запущено")
    
    if worker_pid:
        print(f"✅ Worker:         PID {worker_pid}")
        worker_running = True
        
        if sys.platform == 'win32':
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {worker_pid}', '/FO', 'CSV'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if worker_pid in result.stdout:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 4:
                        mem_usage = parts[4].replace('"', '').strip()
                        print(f"                  Память: {mem_usage}")
            except Exception:
                pass
    else:
        print(f"❌ Worker:         Не запущен")
    
    print("-" * 60)
    
    if app_running and worker_running:
        print("✅ Все сервисы работают нормально")
    elif app_running or worker_running:
        print("⚠️  Часть сервисов не работает")
    else:
        print("❌ Все сервисы остановлены")
    
    return app_running or worker_running

def main():
    if not ensure_pid_dir():
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("❌ Usage: python process_manager.py [start|stop|restart|status]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        logger.info("🎯 Запуск всех сервисов...")
        app_started = start_app()
        if app_started:
            time.sleep(RESTART_DELAY_APP)
            worker_started = start_worker()
            
            if app_started and worker_started:
                logger.info("✅ Все сервисы запущены!")
            elif app_started:
                logger.warning("⚠️  Приложение запущено, но воркер не удалось запустить")
                sys.exit(1)
            else:
                logger.error("❌ Не удалось запустить сервисы")
                sys.exit(1)
        else:
            logger.error("❌ Не удалось запустить приложение")
            sys.exit(1)
        
    elif command == 'stop':
        logger.info("🛑 Остановка всех сервисов...")
        app_stopped = stop_app()
        worker_stopped = stop_worker()
        
        if app_stopped and worker_stopped:
            logger.info("✅ Все сервисы остановлены!")
        elif app_stopped or worker_stopped:
            logger.warning("⚠️  Часть сервисов уже была остановлена")
        else:
            logger.info("ℹ️  Все сервисы уже были остановлены")
        
    elif command == 'restart':
        logger.info("🔄 Перезапуск всех сервисов...")
        stop_app()
        stop_worker()
        time.sleep(RESTART_DELAY_WORKER)
        
        app_started = start_app()
        if app_started:
            time.sleep(RESTART_DELAY_APP)
            start_worker()
            logger.info("✅ Сервисы перезапущены!")
        else:
            logger.error("❌ Не удалось перезапустить приложение")
            sys.exit(1)
        
    elif command == 'status':
        if not show_status():
            sys.exit(1)
    
    elif command == 'clean':
        # Новая команда - очистка stale PID файлов
        logger.info("🧹 Очистка PID файлов...")
        count = 0
        for pid_file in [APP_PID_FILE, WORKER_PID_FILE]:
            if pid_file.exists():
                pid = get_process_by_pid(pid_file)
                if not pid and not pid_file.exists():
                    count += 1
        logger.info(f"✅ Очищено {count} устаревших PID файлов")
        
    else:
        print(f"❌ Неизвестная команда: {command}")
        print("Допустимые команды: start, stop, restart, status, clean")
        sys.exit(1)

if __name__ == '__main__':
    main()
