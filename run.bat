@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [Pakcoy Vision] Lingkungan virtual .venv belum tersedia.
  echo.
  echo Siapkan sekali saja:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\python -m pip install --upgrade pip
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  echo   .venv\Scripts\python -m pip install -r requirements-ml.txt   ^(opsional, untuk YOLO^)
  echo.
  exit /b 1
)

echo [Pakcoy Vision] Menjalankan server di http://127.0.0.1:8000
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
