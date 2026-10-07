@echo off
setlocal
set "PY=%LOCALAPPDATA%\StudyRetriever\runtime\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Study Retriever runtime is not installed. Run install.cmd first.
  exit /b 127
)
set "STUDY_RETRIEVER_HOME=%LOCALAPPDATA%\StudyRetriever"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "DO_NOT_TRACK=1"
"%PY%" -m study_retriever.cli doctor --full --plugin-root "%~dp0"
exit /b %ERRORLEVEL%
