import unittest
from app import app  # Замените на имя вашего основного файла приложения

class TestIndex(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Отключаем CSRF для тестов
        self.client = app.test_client()

    def test_index_get(self):
        """Тест GET-запроса"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Проверяем наличие ключевых элементов на странице
        self.assertIn(b'Cryptocurrency', response.data)  # Замените на ваш текст
        self.assertIn(b'crypto', response.data)  # Проверяем поле выбора криптовалюты

    def test_index_post_valid(self):
        """Тест POST с валидными данными"""
        response = self.client.post('/', data={
            'crypto': 'bitcoin',
            'currency': 'usd'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'BITCOIN/USD', response.data)  # Или другой ожидаемый результат

    def test_index_post_invalid(self):
        """Тест POST с пустыми полями"""
        response = self.client.post('/', data={
            'crypto': '',
            'currency': ''
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Выберите криптовалюту и валюту', response.text)

if __name__ == '__main__':
    unittest.main()