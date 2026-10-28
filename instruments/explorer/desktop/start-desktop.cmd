@echo off
setlocal
set "ELECTRON_RUN_AS_NODE="
if not defined AKD_EXPLORER_URL set "AKD_EXPLORER_URL=http://127.0.0.1:8892/"
if defined AKD_ELECTRON_BIN (
  "%AKD_ELECTRON_BIN%" "%~dp0"
  exit /b
)
if exist "%~dp0node_modules\.bin\electron.cmd" (
  call "%~dp0node_modules\.bin\electron.cmd" "%~dp0"
  exit /b
)
echo Set AKD_ELECTRON_BIN to an installed Electron executable. No runtime is downloaded automatically.
exit /b 1
