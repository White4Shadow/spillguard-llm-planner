param(
    [string]$LlamaBench = ".\tools\llama.cpp-b9804-cuda124\llama-bench.exe",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$GpuLayers = "0,10,20,30,36,40,44,-1",
    [string]$CacheTypes = "q8_0",
    [int]$PromptTokens = 128,
    [int]$GenTokens = 64,
    [int]$Repetitions = 1,
    [string[]]$TensorOverrides = @(),
    [string]$OutDir = ".\benchmarks"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LlamaBench)) {
    throw "llama-bench not found: $LlamaBench"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "model not found: $Model"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$gpuLayerValues = $GpuLayers -split "," | ForEach-Object { [int]$_.Trim() }
$cacheTypeValues = $CacheTypes -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$validCacheTypes = @("f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
foreach ($cacheType in $cacheTypeValues) {
    if ($cacheType -notin $validCacheTypes) {
        throw "invalid cache type '$cacheType'; valid values: $($validCacheTypes -join ', ')"
    }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$modelName = [IO.Path]::GetFileNameWithoutExtension($Model)
$jsonPath = Join-Path $OutDir "$stamp-$modelName-profile.jsonl"
$mdPath = Join-Path $OutDir "$stamp-$modelName-profile.md"

"# llama.cpp fit profile" | Set-Content -LiteralPath $mdPath
"" | Add-Content -LiteralPath $mdPath
"Model: ``$Model``" | Add-Content -LiteralPath $mdPath
"Prompt tokens: $PromptTokens" | Add-Content -LiteralPath $mdPath
"Generation tokens: $GenTokens" | Add-Content -LiteralPath $mdPath
"Repetitions: $Repetitions" | Add-Content -LiteralPath $mdPath
"" | Add-Content -LiteralPath $mdPath

$summary = New-Object System.Collections.Generic.List[object]

foreach ($cacheType in $cacheTypeValues) {
    foreach ($ngl in $gpuLayerValues) {
        $args = @(
            "-m", $Model,
            "-p", "$PromptTokens",
            "-n", "$GenTokens",
            "-r", "$Repetitions",
            "-ngl", "$ngl",
            "-ctk", $cacheType,
            "-ctv", $cacheType,
            "-o", "json"
        )

        if ($TensorOverrides.Count -gt 0) {
            $args += @("-ot", ($TensorOverrides -join ","))
        }

        Write-Host "Running ngl=$ngl cache=$cacheType"
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $raw = & $LlamaBench @args 2>&1 | Out-String
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        $jsonMatch = [regex]::Match($raw, "(?s)\[\s*\{.*\}\s*\]")

        if ($exitCode -ne 0 -or -not $jsonMatch.Success) {
            [pscustomobject]@{
                timestamp = (Get-Date).ToString("o")
                ngl = $ngl
                cache = $cacheType
                ok = $false
                exit_code = $exitCode
                raw = $raw
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $jsonPath
            continue
        }

        $json = $jsonMatch.Value
        $rows = $json | ConvertFrom-Json
        foreach ($row in $rows) {
            $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $jsonPath
            $summary.Add($row) | Out-Null
        }
    }
}

$tgRows = $summary | Where-Object { $_.n_gen -gt 0 } | Sort-Object avg_ts -Descending
$ppRows = $summary | Where-Object { $_.n_prompt -gt 0 } | Sort-Object avg_ts -Descending

"## Best Generation Settings" | Add-Content -LiteralPath $mdPath
"" | Add-Content -LiteralPath $mdPath
$tgRows |
    Select-Object -First 12 n_gpu_layers,type_k,type_v,n_gen,avg_ts,stddev_ts |
    Format-Table -AutoSize |
    Out-String |
    Add-Content -LiteralPath $mdPath

"## Best Prompt Processing Settings" | Add-Content -LiteralPath $mdPath
"" | Add-Content -LiteralPath $mdPath
$ppRows |
    Select-Object -First 12 n_gpu_layers,type_k,type_v,n_prompt,avg_ts,stddev_ts |
    Format-Table -AutoSize |
    Out-String |
    Add-Content -LiteralPath $mdPath

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $mdPath"
