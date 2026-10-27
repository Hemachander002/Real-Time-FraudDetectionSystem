
@echo off
cd /d "E:\fraud detection"

set "PYTHON=E:\fraud detection\venv\python.exe"

echo [1/4] Starting transactions3.py...
start "Script 1 - Transactions" cmd /k ""%PYTHON%" "E:\fraud detection\src\transactions.py""

echo Waiting 20 seconds...
timeout /t 20 /nobreak >nul

echo [2/4] Starting inference_consumer2.py...
start "Script 2 - Inference" cmd /k ""%PYTHON%" "E:\fraud detection\src\inference_consumer.py""

echo Waiting 10 seconds...
timeout /t 10 /nobreak >nul

echo [3/4] Starting kafka_producer3.py...
start "Script 3 - Kafka Producer" cmd /k ""%PYTHON%" "E:\fraud detection\src\kafka_producer.py""

echo Waiting 30 seconds...
timeout /t 30 /nobreak >nul

echo [4/4] Starting outcome_simulator2.py...
start "Script 4 - Outcome Simulator" cmd /k ""%PYTHON%" "E:\fraud detection\src\outcome_simulator.py""

echo All four scripts have been launched.
pause
