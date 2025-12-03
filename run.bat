@echo off
cd /d "C:\cTrader_email"
for /f "delims=" %%a in (.env) do set %%a
"C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe" ctrader.py >> log.txt 2>&1

