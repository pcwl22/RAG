[CmdletBinding()]
param(
    [string]$EnvironmentName = "industrial-rag"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$requirements = Join-Path $projectRoot "requirements-gpu.lock.txt"
$torchRequirements = Join-Path $projectRoot "requirements-torch-cu126.lock.txt"

Write-Host "Installing hash-locked CUDA 12.6 PyTorch into conda env '$EnvironmentName'..."
if (-not (Test-Path -LiteralPath $torchRequirements -PathType Leaf)) {
    throw "Missing CUDA Torch lock: $torchRequirements"
}
conda run -n $EnvironmentName python -m pip install `
    --no-deps --require-hashes `
    --index-url https://download.pytorch.org/whl/cu126 `
    -r $torchRequirements
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }

Write-Host "Installing the remaining hash-locked verified dependencies..."
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Missing GPU dependency lock: $requirements"
}
conda run -n $EnvironmentName python -m pip install --require-hashes -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Host "Installing project metadata without resolving outside the lock..."
conda run -n $EnvironmentName python -m pip install --no-deps -e $projectRoot
if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }

conda run -n $EnvironmentName python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependency graph is inconsistent" }

Write-Host "Running CUDA smoke test..."
conda run -n $EnvironmentName python (Join-Path $PSScriptRoot "check_cuda.py")
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke test failed" }
