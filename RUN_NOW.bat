@echo off
setlocal
cd /d "%~dp0"
echo Starting TextSense OCR with myvenv...
start "" http://127.0.0.1:4444
"C:\Users\MSI\OneDrive\Bureau\talan_rag\myvenv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 4444 --reload
endlocal
