@echo off
REM AI服务启动脚本（Windows）

echo ==========================================
echo Bridge Inspection AI Service
echo ==========================================

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM 检查模型文件
if not exist "weights\best.pt" (
    echo Error: Model weights not found at weights\best.pt
    exit /b 1
)

REM 创建日志目录
if not exist "logs" mkdir logs

REM 设置端口
if "%PORT%"=="" set PORT=5000

echo Starting AI service on port %PORT%...
echo ==========================================

REM 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT% --workers 1 --log-level info

pause
