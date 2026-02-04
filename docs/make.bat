@echo off
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
