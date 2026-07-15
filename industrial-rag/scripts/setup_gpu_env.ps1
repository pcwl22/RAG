[CmdletBinding()]
param(
    [string]$EnvironmentName = "industrial-rag"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $projectRoot "requirements-gpu-verified.txt"

Write-Host "Installing verified CUDA 12.1 PyTorch wheels into conda env '$EnvironmentName'..."
conda run -n $EnvironmentName python -m pip install `
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 `
    --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }

Write-Host "Installing the remaining verified dependencies..."
conda run -n $EnvironmentName python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Host "Running CUDA smoke test..."
conda run -n $EnvironmentName python (Join-Path $PSScriptRoot "check_cuda.py")
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke test failed" }
