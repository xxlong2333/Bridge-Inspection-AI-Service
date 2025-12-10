@echo off
REM 为了避免中文乱码，将控制台编码改为 UTF-8
chcp 65001 >nul

echo ==========================================
echo Bridge Inspection AI Service - 启动中...
echo ==========================================


REM ==========================================
REM Bridge Inspection AI 一键启动脚本（Windows）
REM 功能：
REM 1. 检查 Python 环境
REM 2. 创建并使用本地虚拟环境 venv
REM 3. 安装 requirements.txt 中的依赖
REM 4. 检查模型权重 weights\best.pt
REM 5. 启动 uvicorn 服务
REM ==========================================

echo ==========================================
echo Bridge Inspection AI Service - Startup
echo ==========================================

REM ---------- 1. 检查 Python 是否可用 ----------
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并加入系統 PATH
    goto END
)

REM ---------- 2. 创建 / 检查虚拟环境 ----------
set "VENV_DIR=venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [信息] 未检测到虚拟环境，正在创建 venv ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        goto END
    )
) else (
    echo [信息] 已检测到虚拟环境：%VENV_DIR%
)

REM ---------- 3. 激活虚拟环境 ----------
echo [信息] 正在激活虚拟环境 ...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    goto END
)

REM ---------- 4. 安装依赖 ----------
if exist "requirements.txt" (
    echo [信息] 正在安装/更新依赖（pip install -r requirements.txt）...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查 requirements.txt
        goto END
    )
) else (
    echo [警告] 当前目录未找到 requirements.txt，跳过依赖安装
)

REM ---------- 5. 检查模型文件 ----------
if not exist "weights\best.pt" (
    echo [错误] 未找到模型权重文件：weights\best.pt
    echo 请将模型权重放到 weights\best.pt 后再运行本脚本。
    goto END
) else if not exist "weights\car-best.pt" (
    echo [错误] 未找到模型权重文件：weights\car-best.pt
    echo 请将模型权重放到 weights\car-best.pt 后再运行本脚本。
    goto END
) else (
    echo [信息] 模型权重已找到!
)

REM ---------- 6. 创建日志目录 ----------
if not exist "logs" (
    echo [信息] 正在创建日志目录 logs ...
    mkdir logs
)

REM ---------- 7. 设置端口 ----------
if "%PORT%"=="" set PORT=5000

echo.
echo ==========================================
echo 即将启动 AI 服务...
echo 监听地址: 0.0.0.0:%PORT%
echo 使用虚拟环境: %VENV_DIR%
echo ==========================================
echo.

REM ---------- 8. 启动 uvicorn 服务 ----------
python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT% --workers 1 --log-level info

:END
echo.
echo 按任意键退出...
pause >nul
