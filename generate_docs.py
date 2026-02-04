#!/usr/bin/env python3
"""Генератор полной документации с графами вызовов для Crypto Analytics Platform"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path
from datetime import datetime
import ast
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ДЛЯ АНАЛИЗА КОДА
# ============================================================================

class CallGraphGenerator:
    """Генератор графов вызовов функций из Python кода с использованием AST"""
    
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.graph = nx.DiGraph()
        
    def analyze_file(self, filepath: str) -> dict:
        """Анализирует файл Python, извлекая функции, классы и их вызовы"""
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        functions = {}
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                calls = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Name):
                            calls.append(subnode.func.id)
                        elif isinstance(subnode.func, ast.Attribute):
                            calls.append(subnode.func.attr)
                
                functions[node.name] = {
                    'type': 'function',
                    'line': node.lineno,
                    'calls': calls
                }
            
            elif isinstance(node, ast.ClassDef):
                class_methods = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_calls = []
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Call):
                                if isinstance(subnode.func, ast.Name):
                                    method_calls.append(subnode.func.id)
                                elif isinstance(subnode.func, ast.Attribute):
                                    method_calls.append(subnode.func.attr)
                        
                        class_methods[item.name] = {
                            'line': item.lineno,
                            'calls': method_calls
                        }
                
                classes[node.name] = {
                    'type': 'class',
                    'line': node.lineno,
                    'methods': class_methods
                }
        
        return {
            'functions': functions,
            'classes': classes,
            'filename': os.path.basename(filepath)
        }
    
    def build_graph(self):
        """Строит граф зависимостей на основе анализа кода"""
        analysis = self.analyze_file(self.source_path)
        
        for func_name, func_info in analysis['functions'].items():
            self.graph.add_node(
                f"function:{func_name}",
                type='function',
                line=func_info['line'],
                module='app'
            )
            
            for call in func_info['calls']:
                if call in analysis['functions']:
                    self.graph.add_edge(
                        f"function:{func_name}",
                        f"function:{call}"
                    )
        
        for class_name, class_info in analysis['classes'].items():
            self.graph.add_node(
                f"class:{class_name}",
                type='class',
                line=class_info['line'],
                module='app'
            )
            
            for method_name, method_info in class_info['methods'].items():
                node_id = f"class:{class_name}.{method_name}"
                self.graph.add_node(
                    node_id,
                    type='method',
                    line=method_info['line'],
                    parent=class_name
                )
                
                self.graph.add_edge(f"class:{class_name}", node_id)
                
                for call in method_info['calls']:
                    if call in analysis['functions']:
                        self.graph.add_edge(node_id, f"function:{call}")
        
        return self.graph
    
    def generate_visualization(self, output_path: str, format='png'):
        """Создает визуализацию графа вызовов"""
        if len(self.graph.nodes()) == 0:
            return
        
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(self.graph, k=1.5, iterations=100, seed=42)
        
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            color_map = {
                'function': 'lightblue',
                'class': 'lightgreen', 
                'method': 'lightcoral'
            }
            size_map = {
                'function': 2000,
                'class': 3000,
                'method': 1500
            }
            
            node_colors.append(color_map.get(node_type, 'gray'))
            node_sizes.append(size_map.get(node_type, 1000))
        
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=12,
            width=1,
            alpha=0.8
        )
        
        plt.title(f'Граф вызовов: {os.path.basename(self.source_path)}', fontsize=16, pad=20)
        plt.savefig(output_path, format=format, dpi=100, bbox_inches='tight')
        plt.close()
    
    def generate_json_report(self, output_path: str):
        """Генерирует структурированный отчет о графе в формате JSON"""
        report = {
            'metadata': {
                'source_file': self.source_path,
                'total_nodes': len(self.graph.nodes()),
                'total_edges': len(self.graph.edges()),
                'generated_at': datetime.now().isoformat()
            },
            'statistics': {
                'functions': 0,
                'classes': 0,
                'methods': 0,
                'orphan_nodes': 0
            },
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            node_info = {'id': node, **attrs}
            report['nodes'].append(node_info)
            
            node_type = attrs.get('type', '')
            if node_type == 'function':
                report['statistics']['functions'] += 1
            elif node_type == 'class':
                report['statistics']['classes'] += 1
            elif node_type == 'method':
                report['statistics']['methods'] += 1
        
        for source, target in self.graph.edges():
            report['edges'].append({'source': source, 'target': target})
        
        orphan_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        report['statistics']['orphan_nodes'] = len(orphan_nodes)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


# ============================================================================
# ОСНОВНЫЕ ФУНКЦИИ ГЕНЕРАЦИИ ДОКУМЕНТАЦИИ
# ============================================================================

def setup_environment():
    """Создает структуру директорий и базовые файлы для Sphinx"""
    
    docs_dir = Path('docs')
    source_dir = docs_dir / 'source'
    
    if (source_dir / 'conf.py').exists():
        print("📁 Конфигурация Sphinx уже существует")
        return
    
    directories = [
        docs_dir,
        source_dir,
        source_dir / '_static',
        source_dir / '_extensions',
        source_dir / '_static/callgraphs',
        source_dir / '_static/analysis'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Создаем Makefile для Linux/Mac и batch файл для Windows
    if platform.system() == 'Windows':
        makefile_content = '''@echo off
echo Минимальный Makefile для Sphinx в Windows

set SPHINXOPTS=
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS%
goto :eof

:clean
if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
goto :eof

:html
%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%/html" %SPHINXOPTS%
echo.
echo Документация построена. Откройте %BUILDDIR%/html/index.html
goto :eof
'''
        makefile_path = docs_dir / 'make.bat'
    else:
        makefile_content = '''# Минимальный Makefile для Sphinx

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html

help:
\t@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
\trm -rf $(BUILDDIR)/*

html:
\t$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
\t@echo
\t@echo "Документация построена. Откройте $(BUILDDIR)/html/index.html"
'''
        makefile_path = docs_dir / 'Makefile'
    
    with open(makefile_path, 'w', encoding='utf-8') as f:
        f.write(makefile_content)
    
    requirements = """sphinx>=7.2.0
sphinx-autodoc-typehints>=1.25.0
sphinx-rtd-theme>=2.0.0
sphinx-autoapi>=3.0.0
sphinxcontrib-mermaid>=0.9.0
networkx>=3.0
matplotlib>=3.7.0
"""
    
    with open(docs_dir / 'requirements-docs.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("✅ Структура документации создана")


def create_sphinx_config():
    """Генерирует конфигурационный файл для Sphinx"""
    
    conf_content = '''"""
Конфигурация Sphinx для Crypto Analytics Platform
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../..'))

project = 'Crypto Analytics Platform'
author = 'Романов Е.В.'
copyright = f'{datetime.now().year}, {author}'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.mermaid',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autosummary_generate = True
autoclass_content = 'both'
add_module_names = False

language = 'ru'

exclude_patterns = []
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True
}
'''
    
    source_dir = Path('docs/source')
    with open(source_dir / 'conf.py', 'w', encoding='utf-8') as f:
        f.write(conf_content)
    
    print("✅ Конфигурация Sphinx создана")


def analyze_application():
    """Анализирует структуру приложения и генерирует графы вызовов"""
    
    print("🔍 Анализ структуры приложения...")
    
    app_path = Path('app.py')
    if not app_path.exists():
        print(f"⚠️  Файл {app_path} не найден")
        return
    
    try:
        generator = CallGraphGenerator(str(app_path))
        graph = generator.build_graph()
        
        output_dir = Path('docs/source/_static/callgraphs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        png_path = output_dir / 'full_callgraph.png'
        generator.generate_visualization(str(png_path))
        print(f"✅ Граф сохранен: {png_path}")
        
        json_path = output_dir / 'callgraph_report.json'
        report = generator.generate_json_report(str(json_path))
        
        stats = report['statistics']
        print(f"📊 Статистика анализа:")
        print(f"   • Узлы: {len(report['nodes'])}")
        print(f"   • Связи: {len(report['edges'])}")
        print(f"   • Функции: {stats['functions']}")
        print(f"   • Классы: {stats['classes']}")
        print(f"   • Методы: {stats['methods']}")
        
        generate_module_analysis(report)
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")


def generate_module_analysis(report):
    """Анализирует функции по категориям"""
    
    categories = {
        'database': ['db', 'query', 'session', 'select', 'insert', 'update'],
        'api': ['api', 'request', 'response', 'endpoint', 'get_', 'post_'],
        'routes': ['route', 'index', 'chart', 'correlation', 'auth'],
        'utils': ['log', 'cache', 'config', 'helper', 'util', 'load_'],
        'analysis': ['calculate', 'correlation', 'stat', 'plot', 'graph', 'analyze']
    }
    
    functions_by_category = {cat: [] for cat in categories.keys()}
    
    for node in report['nodes']:
        if node.get('type') == 'function':
            func_name = node['id'].replace('function:', '')
            category = 'utils'
            
            for cat, keywords in categories.items():
                if any(keyword in func_name.lower() for keyword in keywords):
                    category = cat
                    break
            
            functions_by_category[category].append(func_name)
    
    connected_functions = []
    for node in report['nodes']:
        if node.get('type') == 'function':
            func_id = node['id']
            connections = sum(1 for edge in report['edges'] 
                            if edge['source'] == func_id or edge['target'] == func_id)
            connected_functions.append((func_id.replace('function:', ''), connections))
    
    connected_functions.sort(key=lambda x: x[1], reverse=True)
    
    analysis_data = {
        'total_functions': report['statistics']['functions'],
        'functions_by_category': functions_by_category,
        'most_connected_functions': connected_functions[:10],
        'generated_at': datetime.now().isoformat()
    }
    
    analysis_dir = Path('docs/source/_static/analysis')
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_dir / 'module_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Анализ модулей сохранен")


def create_rst_files():
    """Создает полный набор RST файлов для документации"""
    
    source_dir = Path('docs/source')
    
    rst_templates = {
        'index.rst': '''Документация Crypto Analytics Platform
=======================================

.. toctree::
   :maxdepth: 3
   :caption: Содержание:
   
   overview
   callgraphs
   architecture
   usage

Быстрый старт
-------------

Веб-приложение для анализа криптовалютных данных с Flask-бэкендом.

Установка и запуск
------------------

.. code-block:: bash

   git clone <your-repo>
   cd crypto_app
   
   pip install -r requirements.txt
   python app.py
   
   http://localhost:5000
''',
        
        'overview.rst': '''Обзор приложения
==================

Назначение
----------

Crypto Analytics Platform — веб-приложение для анализа криптовалютных рынков.

Технологический стек
--------------------

* **Backend**: Flask, SQLAlchemy
* **Frontend**: HTML, CSS, JavaScript, Bootstrap
* **База данных**: MySQL/PostgreSQL
* **Внешние API**: CoinGecko, Binance, Telegram
''',
        
        'callgraphs.rst': '''Графы вызовов функций
======================

Полный граф вызовов
-------------------

.. image:: _static/callgraphs/full_callgraph.png
   :width: 100%
   :alt: Полный граф вызовов функций
   :align: center

Статистика анализа
------------------

* `callgraph_report.json <_static/callgraphs/callgraph_report.json>`_
* `module_analysis.json <_static/analysis/module_analysis.json>`_
''',
        
        'architecture.rst': '''Архитектура приложения
=======================

Трехзвенная архитектура
-----------------------

1. **Презентационный слой** — Flask маршруты
2. **Бизнес-логика** — функции обработки данных
3. **Слой данных** — модели SQLAlchemy и внешние API
''',
        
        'usage.rst': '''Руководство по использованию
=======================

Установка
---------

1. Клонирование репозитория
2. Создание виртуального окружения
3. Установка зависимостей
4. Настройка базы данных
5. Запуск приложения

API Endpoints
-------------

* ``GET /`` — Главная страница
* ``POST /telegram-auth`` — Авторизация через Telegram
* ``GET /chart`` — Страница графиков
* ``GET /correlation`` — Анализ корреляций
* ``GET /api/status`` — Статус системы
'''
    }
    
    for filename, content in rst_templates.items():
        with open(source_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("✅ RST файлы созданы")


def build_documentation():
    """Запускает сборку HTML документации с помощью Sphinx"""
    
    print("🔨 Сборка документации...")
    
    try:
        # Проверяем установлен ли sphinx-build
        try:
            subprocess.run(['sphinx-build', '--version'], 
                         capture_output=True, check=True)
            sphinx_installed = True
        except:
            sphinx_installed = False
        
        if not sphinx_installed:
            print("📦 Установка Sphinx...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'sphinx', 'sphinx-rtd-theme', 'sphinxcontrib-mermaid'],
                         capture_output=True)
        
        # Прямой вызов sphinx-build вместо make
        os.chdir('docs')
        
        # Очистка предыдущей сборки
        build_dir = Path('build')
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        
        # Сборка документации
        print("🏗️  Генерация HTML документации...")
        result = subprocess.run([
            'sphinx-build', '-b', 'html', 
            'source', 'build/html',
            '-q'  # Тихий режим
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*50)
            print("✅ Документация успешно собрана!")
            
            index_path = Path('build/html/index.html').absolute()
            if index_path.exists():
                print(f"📁 Файл: {index_path}")
                
                # Попытка открыть в браузере (только Windows)
                if platform.system() == 'Windows':
                    try:
                        os.startfile(index_path)
                        print("🌐 Открываю в браузере...")
                    except:
                        pass
            print("="*50)
        else:
            print(f"❌ Ошибка сборки: {result.stderr[:500]}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        os.chdir('..')


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основной рабочий процесс генерации документации"""
    
    print("\n🚀 Запуск генерации документации")
    print("="*50)
    
    try:
        setup_environment()
        create_sphinx_config()
        analyze_application()
        create_rst_files()
        build_documentation()
        
    except KeyboardInterrupt:
        print("\n⏹️  Генерация прервана")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*50)
    print("🎉 Генерация завершена!\n")


if __name__ == '__main__':
    main()