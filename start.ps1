$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if (-not (Test-Path "frontend\dist\index.html")) {
  throw "Frontend is not built. Run .\setup.ps1 first."
}
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (Test-Path $BundledPython) {
  $Python = $BundledPython
} elseif ($PythonCommand) {
  $Python = $PythonCommand.Source
} else {
  throw "Python was not found. Install Python 3.11 or 3.12 and run this script again."
}
& $Python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
