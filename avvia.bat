@echo off
setlocal

cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
)

if not exist "data" mkdir "data"
if not exist "chroma_db" mkdir "chroma_db"

set "MISTRAL_API_KEY_VALUE="
for /f "tokens=1,* delims==" %%A in ('findstr /B "MISTRAL_API_KEY=" ".env"') do set "MISTRAL_API_KEY_VALUE=%%B"
if "%MISTRAL_API_KEY_VALUE%"=="" (
    echo.
    echo MISTRAL_API_KEY manca nel file .env.
    echo Apri .env e inserisci la tua chiave Mistral, poi rilancia questo file.
    echo.
    pause
    exit /b 1
)

dir /b "data\*.pdf" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Nessun PDF trovato nella cartella data.
    echo Metti almeno un PDF in data e rilancia questo file.
    echo.
    pause
    exit /b 1
)

docker compose version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Docker Compose non risulta disponibile.
    echo Avvia Docker Desktop o installa Docker, poi rilancia questo file.
    echo.
    pause
    exit /b 1
)

echo.
echo Avvio RAG in Docker...
echo Interfaccia web: http://localhost:8501
echo.
start "" "http://localhost:8501"
docker compose up --build

echo.
pause
