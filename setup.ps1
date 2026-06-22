param([switch]$ForceBuild)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BundledRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies"
$BundledPython = Join-Path $BundledRoot "python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (Test-Path $BundledPython) {
  $Python = $BundledPython
} elseif ($PythonCommand) {
  $Python = $PythonCommand.Source
} else {
  throw "Python was not found. Install Python 3.11 or 3.12 and run this script again."
}

& $Python -c "import cv2, fastapi, httpx, mediapipe, numpy, pytest, uvicorn; assert mediapipe.__version__ == '0.10.21'"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing Python dependencies..."
  & $Python -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
} else {
  Write-Host "Python dependencies are already available."
}

if ((Test-Path "frontend\dist\index.html") -and -not $ForceBuild) {
  Write-Host "The prebuilt dashboard is already available."
} else {
  Push-Location frontend
  try {
    $NpmCommand = Get-Command npm -ErrorAction SilentlyContinue
    if ($NpmCommand) {
      & $NpmCommand.Source install
      if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
      & $NpmCommand.Source run build
      if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } else {
      $Pnpm = Join-Path $BundledRoot "bin\pnpm.cmd"
      $NodeDirectory = Join-Path $BundledRoot "node\bin"
      if (-not (Test-Path $Pnpm) -or -not (Test-Path (Join-Path $NodeDirectory "node.exe"))) {
        throw "Node.js/npm was not found. Install Node.js 20+ and run this script again."
      }
      $env:PATH = "$NodeDirectory;$env:PATH"
      $env:CI = "true"
      & $Pnpm install
      if ($LASTEXITCODE -ne 0) { throw "pnpm install failed." }
      & $Pnpm run build
      if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
  } finally {
    Pop-Location
  }
}
Write-Host "Setup complete. Run .\start.ps1"
