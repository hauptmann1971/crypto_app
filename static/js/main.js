/**
 * Crypto App - Основные JavaScript утилиты
 */

// Получение CSRF токена из meta тега
function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

// AJAX запрос с CSRF токеном
async function ajaxRequest(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        }
    };
    
    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Ошибка запроса');
        }
        
        return result;
    } catch (error) {
        console.error('AJAX Error:', error);
        throw error;
    }
}

// Показ уведомления
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.main-content');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Автоматическое скрытие через 5 секунд
        setTimeout(() => {
            alertDiv.style.opacity = '0';
            alertDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => alertDiv.remove(), 300);
        }, 5000);
    }
}

// Форматирование числа как валюты
function formatCurrency(value, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: 2,
        maximumFractionDigits: 8
    }).format(value);
}

// Форматирование даты
function formatDate(timestamp) {
    return new Date(timestamp * 1000).toLocaleString('ru-RU', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Дебаунс функция для оптимизации частых вызовов
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Проверка статуса процесса
async function checkProcessStatus(requestId) {
    try {
        const response = await ajaxRequest(`/request-status/${requestId}`);
        
        if (response.success) {
            if (response.status === 'finished') {
                return {
                    success: true,
                    rate: response.rate,
                    crypto: response.crypto,
                    currency: response.currency
                };
            } else if (response.status === 'error') {
                return {
                    success: false,
                    error: response.error
                };
            }
            // Ещё обрабатывается
            return {
                success: false,
                processing: true
            };
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
    
    return {
        success: false,
        error: 'Не удалось проверить статус'
    };
}

// Ожидание завершения запроса с поллингом
async function waitForRequestCompletion(requestId, maxAttempts = 60, interval = 1000) {
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        const status = await checkProcessStatus(requestId);
        
        if (!status.processing) {
            return status;
        }
        
        await new Promise(resolve => setTimeout(resolve, interval));
        attempts++;
    }
    
    throw new Error('Превышено время ожидания запроса');
}

// Инициализация tooltips
function initTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', (e) => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = element.getAttribute('data-tooltip');
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                font-size: 0.875rem;
                z-index: 1000;
                pointer-events: none;
            `;
            
            document.body.appendChild(tooltip);
            
            const rect = element.getBoundingClientRect();
            tooltip.style.top = `${rect.top - tooltip.offsetHeight - 5}px`;
            tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;
            
            element._tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', () => {
            if (element._tooltip) {
                element._tooltip.remove();
                element._tooltip = null;
            }
        });
    });
}

// Автоинициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    initTooltips();
    
    // Автоматическое скрытие алертов через 5 секунд
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-20px)';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });
});

// Экспорт функций для глобального доступа
window.CryptoApp = {
    ajaxRequest,
    showNotification,
    formatCurrency,
    formatDate,
    checkProcessStatus,
    waitForRequestCompletion,
    getCSRFToken
};
