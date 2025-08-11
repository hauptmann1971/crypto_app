import pytest
from app import create_app
import mysql.connector
from datetime import datetime, timedelta
import time
from config import *

# Конфигурация для тестовой БД
TEST_DB_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME
}

@pytest.fixture(scope='module')
def test_app():
    """Создаем тестовое приложение с реальной БД"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['DB_CONFIG'] = TEST_DB_CONFIG
    
    # Инициализируем тестовую БД
    with app.app_context():
        conn = mysql.connector.connect(**TEST_DB_CONFIG)
        cursor = conn.cursor()
        
        # Создаем тестовые таблицы
        cursor.execute("DROP TABLE IF EXISTS crypto_rates")
        cursor.execute("DROP TABLE IF EXISTS app_logs")
        
        cursor.execute("""
            CREATE TABLE crypto_rates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                crypto VARCHAR(50) NOT NULL,
                currency VARCHAR(10) NOT NULL,
                rate FLOAT NOT NULL,
                source VARCHAR(200) NOT NULL,
                epoch_time BIGINT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE app_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                message VARCHAR(500) NOT NULL,
                level VARCHAR(20) NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
    yield app
    
    # Очистка после тестов
    with app.app_context():
        conn = mysql.connector.connect(**TEST_DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS crypto_rates")
        cursor.execute("DROP TABLE IF EXISTS app_logs")
        conn.commit()
        cursor.close()
        conn.close()

@pytest.fixture
def client(test_app):
    """Тестовый клиент"""
    return test_app.test_client()

def test_index_get_real(client):
    """Тест GET / с реальными данными"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Crypto Exchange Rates' in response.data
    
    # Проверяем, что основные криптовалюты есть в списке
    assert b'bitcoin' in response.data
    assert b'ethereum' in response.data
    assert b'binancecoin' in response.data

def test_index_post_real(client):
    """Тест POST / с реальным запросом к API"""
    # Делаем запрос с реальными данными
    response = client.post('/', data={
        'crypto': 'bitcoin',
        'currency': 'usd'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    
    # Проверяем, что получили какой-то курс
    assert b'bitcoin/USD:' in response.data
    
    # Проверяем запись в БД
    conn = mysql.connector.connect(**TEST_DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crypto_rates WHERE crypto='bitcoin' AND currency='usd'")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    
    assert len(result) >= 1  # Должна быть хотя бы одна запись
    assert isinstance(result[0][3], float)  # Проверяем что rate это число

def test_show_crypto_rates_real(client):
    """Тест /crypto с реальными данными из БД"""
    # Сначала добавляем тестовые данные
    conn = mysql.connector.connect(**TEST_DB_CONFIG)
    cursor = conn.cursor()
    test_time = int(time.time())
    
    cursor.execute("""
        INSERT INTO crypto_rates 
        (crypto, currency, rate, source, epoch_time)
        VALUES (%s, %s, %s, %s, %s)
    """, ('ethereum', 'eur', 2500.0, 'test_source', test_time))
    conn.commit()
    cursor.close()
    conn.close()
    
    # Теперь проверяем вывод
    response = client.get('/crypto')
    assert response.status_code == 200
    
     # проверяем наличие добавленных данных
    assert b'ethereum' in response.data
    assert b'eur' in response.data

def test_logging_real(client):
    """Тест работы системы логирования"""
    # Вызываем метод, который должен логировать
    response = client.post('/', data={
        'crypto': 'ethereum',
        'currency': 'gbp'
    }, follow_redirects=True)
    
    # Проверяем запись в логах
    conn = mysql.connector.connect(**TEST_DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM app_logs WHERE message LIKE '%ethereum/gbp%'")
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    
    assert len(logs) >= 1  # Должна быть хотя бы одна запись

def test_error_handling_real(client):
    """Тест обработки ошибок с несуществующей криптовалютой"""
    response = client.post('/', data={
        'crypto': 'nonexistent_coin',
        'currency': 'usd'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b'Не удалось получить курс' in response.data
    
    # Проверяем запись об ошибке в логах
    conn = mysql.connector.connect(**TEST_DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM app_logs WHERE message LIKE '%nonexistent_coin%'")
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    
    assert len(logs) >= 1