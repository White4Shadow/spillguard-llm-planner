param(
    [string]$LlamaDir = ".\tools\llama.cpp-src",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [int]$GpuLayers = 12,
    [int]$NewTokens = 4,
    [int]$MigrateAt = 0,
    [double]$MinGpuDeltaMb = 1.0,
    [string]$CudaPath = "",
    [string]$CudaArchitecture = "86",
    [string]$Prompt = "Hello",
    [string]$BuildDir = "",
    [string]$CMake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Find-CudaPath([string]$RequestedPath) {
    if ($RequestedPath) {
        return Resolve-RequiredPath $RequestedPath "CUDA Toolkit"
    }

    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        return (Resolve-Path -LiteralPath (Join-Path $nvcc.Source "..\..")).Path
    }

    $cudaRoots = Get-ChildItem -Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($root in $cudaRoots) {
        if (Test-Path -LiteralPath (Join-Path $root.FullName "bin\nvcc.exe")) {
            return $root.FullName
        }
    }

    throw "CUDA Toolkit nvcc was not found. Install NVIDIA CUDA Toolkit before running the CUDA migration proof."
}

if (-not (Test-Path -LiteralPath $CMake)) {
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmakeCommand) {
        throw "cmake not found. Pass -CMake or install CMake."
    }
    $CMake = $cmakeCommand.Source
}

$llamaPath = Resolve-RequiredPath $LlamaDir "patched llama.cpp source"
$modelPath = Resolve-RequiredPath $Model "model"
$cudaRoot = Find-CudaPath $CudaPath
$nvccPath = Resolve-RequiredPath (Join-Path $cudaRoot "bin\nvcc.exe") "nvcc"

$env:PATH = "$cudaRoot\bin;$cudaRoot\bin\x64;$env:PATH"
$env:CUDA_PATH = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot
$env:CudaToolkitDir = "$cudaRoot\"
$versionKey = Split-Path -Leaf $cudaRoot
if ($versionKey -match "^v(\d+)\.(\d+)") {
    Set-Item -Path "Env:CUDA_PATH_V$($matches[1])_$($matches[2])" -Value $cudaRoot
}

if ($BuildDir) {
    if ([System.IO.Path]::IsPathRooted($BuildDir)) {
        $buildPath = $BuildDir
    } else {
        $buildPath = Join-Path (Get-Location).Path $BuildDir
    }
} else {
    $buildPath = Join-Path $llamaPath "build-live-migration-cuda"
}

Write-Host "Configuring CUDA build in $buildPath"
Write-Host "Using CUDA Toolkit $cudaRoot"
Write-Host "Using nvcc $nvccPath"
& $CMake `
    -S $llamaPath `
    -B $buildPath `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -T "cuda=$cudaRoot" `
    -DGGML_CUDA=ON `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture" `
    -DGGML_NATIVE=OFF `
    -DLLAMA_BUILD_EXAMPLES=ON `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_TESTS=OFF
if ($LASTEXITCODE -ne 0) {
    throw "CUDA CMake configure failed with exit code $LASTEXITCODE"
}

Write-Host "Building llama-live-migration-probe"
& $CMake --build $buildPath --config Release --target llama-live-migration-probe
if ($LASTEXITCODE -ne 0) {
    throw "CUDA probe build failed with exit code $LASTEXITCODE"
}

$probe = Join-Path $buildPath "bin\Release\llama-live-migration-probe.exe"
$probe = Resolve-RequiredPath $probe "llama-live-migration-probe"
$minDeltaText = [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $MinGpuDeltaMb)

$env:LLAMA_EXPERIMENTAL_LAYER_BUFFERS = "1"

Write-Host "Running strict GPU memory-release probe"
& $probe `
    -m $modelPath `
    -ngl $GpuLayers `
    -n $NewTokens `
    --migrate-at $MigrateAt `
    --target cpu `
    --require-gpu `
    --expect-gpu-delta-mb $minDeltaText `
    $Prompt
if ($LASTEXITCODE -ne 0) {
    throw "strict GPU memory-release probe failed with exit code $LASTEXITCODE"
}

Write-Host "CUDA live migration proof passed."
