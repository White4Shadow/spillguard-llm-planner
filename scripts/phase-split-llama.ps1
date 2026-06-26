param(
    [string]$Server = ".\tools\llama.cpp-b9804-cuda124\llama-server.exe",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$PromptText = "",
    [string]$PromptFile = "",
    [string]$PrefillGpuLayers = "0",
    [string]$DecodeGpuLayers = "-1",
    [string]$CacheType = "q8_0",
    [int]$ContextSize = 2048,
    [int]$NewTokens = 32,
    [ValidateSet("same", "empty")]
    [string]$DecodePromptMode = "same",
    [int]$PortBase = 18100,
    [switch]$CompareCold,
    [string]$OutDir = ".\benchmarks",
    [string]$SlotDir = ".\phase-cache",
    [string]$SlotFile = "",
    [switch]$UseExistingSlot
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Stop-ServerProcess($Process) {
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force
        Wait-Process -Id $Process.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
}

function Wait-ServerReady([int]$Port, [int]$TimeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 1 | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "server on port $Port did not become ready within $TimeoutSeconds seconds"
}

function Start-LlamaServer(
    [string]$ServerPath,
    [string]$ModelPath,
    [int]$Port,
    [string]$GpuLayers,
    [string]$CacheType,
    [int]$ContextSize,
    [string]$SlotDir,
    [string]$LogPrefix
) {
    $stdout = Join-Path $OutDir "$LogPrefix.out.log"
    $stderr = Join-Path $OutDir "$LogPrefix.err.log"
    $args = @(
        "-m", $ModelPath,
        "--host", "127.0.0.1",
        "--port", "$Port",
        "-ngl", "$GpuLayers",
        "-ctk", $CacheType,
        "-ctv", $CacheType,
        "-c", "$ContextSize",
        "-np", "1",
        "--slot-save-path", $SlotDir,
        "--no-warmup"
    )

    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $ServerPath -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -ArgumentList $args
    Wait-ServerReady -Port $Port
    $watch.Stop()

    return [pscustomobject]@{
        process = $process
        ready_ms = [math]::Round($watch.Elapsed.TotalMilliseconds, 3)
        stdout = $stdout
        stderr = $stderr
    }
}

function Invoke-JsonPost([string]$Uri, [hashtable]$Body, [int]$TimeoutSeconds = 600) {
    $json = $Body | ConvertTo-Json -Depth 12
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri $Uri -Method Post -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSeconds
    $watch.Stop()
    return [pscustomobject]@{
        response = $response
        elapsed_ms = [math]::Round($watch.Elapsed.TotalMilliseconds, 3)
    }
}

$serverPath = Resolve-RequiredPath $Server "llama-server"
$modelPath = Resolve-RequiredPath $Model "model"
New-Item -ItemType Directory -Force -Path $OutDir, $SlotDir | Out-Null
$slotDirPath = (Resolve-Path -LiteralPath $SlotDir).Path

if ($PromptFile) {
    $prompt = Get-Content -LiteralPath (Resolve-RequiredPath $PromptFile "prompt file") -Raw
} elseif ($PromptText) {
    $prompt = $PromptText
} else {
    $prompt = "Phase split inference test. " * 300
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ($SlotFile) {
    if ([System.IO.Path]::IsPathRooted($SlotFile)) {
        $slotFile = [System.IO.Path]::GetFileName($SlotFile)
    } else {
        $slotFile = $SlotFile
    }
} else {
    $slotFile = "phase-split-$stamp.bin"
}

if ($UseExistingSlot) {
    $existingSlot = Join-Path $slotDirPath $slotFile
    if (-not (Test-Path -LiteralPath $existingSlot)) {
        throw "existing slot file not found under SlotDir: $existingSlot"
    }
}

$resultPath = Join-Path $OutDir "$stamp-phase-split-result.json"
$summaryPath = Join-Path $OutDir "$stamp-phase-split-summary.md"
$prefillServer = $null
$decodeServer = $null
$coldServer = $null
$prefillReadyMs = 0
$prefill = $null
$save = $null

try {
    if (-not $UseExistingSlot) {
        $prefillServer = Start-LlamaServer -ServerPath $serverPath -ModelPath $modelPath -Port $PortBase `
            -GpuLayers $PrefillGpuLayers -CacheType $CacheType -ContextSize $ContextSize -SlotDir $slotDirPath `
            -LogPrefix "$stamp-prefill"
        $prefillReadyMs = $prefillServer.ready_ms

        $prefill = Invoke-JsonPost -Uri "http://127.0.0.1:$PortBase/completion" -Body @{
            prompt = $prompt
            n_predict = 0
            id_slot = 0
            cache_prompt = $true
            response_fields = @("tokens_evaluated", "tokens_cached", "timings")
        }

        $save = Invoke-JsonPost -Uri "http://127.0.0.1:$PortBase/slots/0?action=save" -Body @{
            filename = $slotFile
        }

        Stop-ServerProcess $prefillServer.process
        $prefillServer = $null
    }

    $decodePort = $PortBase + 1
    $decodeServer = Start-LlamaServer -ServerPath $serverPath -ModelPath $modelPath -Port $decodePort `
        -GpuLayers $DecodeGpuLayers -CacheType $CacheType -ContextSize $ContextSize -SlotDir $slotDirPath `
        -LogPrefix "$stamp-decode"
    $decodeReadyMs = $decodeServer.ready_ms

    $restore = Invoke-JsonPost -Uri "http://127.0.0.1:$decodePort/slots/0?action=restore" -Body @{
        filename = $slotFile
    }

    $decode = Invoke-JsonPost -Uri "http://127.0.0.1:$decodePort/completion" -Body @{
        prompt = if ($DecodePromptMode -eq "empty") { "" } else { $prompt }
        n_predict = $NewTokens
        id_slot = 0
        cache_prompt = $true
        temperature = 0
        response_fields = @("content", "tokens_evaluated", "tokens_cached", "timings")
    }

    Stop-ServerProcess $decodeServer.process
    $decodeServer = $null

    $cold = $null
    if ($CompareCold) {
        $coldPort = $PortBase + 2
        $coldServer = Start-LlamaServer -ServerPath $serverPath -ModelPath $modelPath -Port $coldPort `
            -GpuLayers $DecodeGpuLayers -CacheType $CacheType -ContextSize $ContextSize -SlotDir $slotDirPath `
            -LogPrefix "$stamp-cold"
        $coldReadyMs = $coldServer.ready_ms

        $cold = Invoke-JsonPost -Uri "http://127.0.0.1:$coldPort/completion" -Body @{
            prompt = $prompt
            n_predict = $NewTokens
            id_slot = 0
            cache_prompt = $true
            temperature = 0
            response_fields = @("content", "tokens_evaluated", "tokens_cached", "timings")
        }

        Stop-ServerProcess $coldServer.process
        $coldServer = $null
    }

    $phaseOneOffMs = $decodeReadyMs + $restore.elapsed_ms + $decode.elapsed_ms
    if (-not $UseExistingSlot) {
        $phaseOneOffMs += $prefillReadyMs + $prefill.elapsed_ms + $save.elapsed_ms
    }
    $coldOneOffMs = $null
    $cachedDecodeSavingMs = $null
    $breakEvenLoadedReuses = $null
    if ($cold) {
        $coldOneOffMs = $coldReadyMs + $cold.elapsed_ms
        $cachedDecodeSavingMs = $cold.elapsed_ms - $decode.elapsed_ms
        if ($cachedDecodeSavingMs -gt 0) {
            if ($UseExistingSlot) {
                $breakEvenLoadedReuses = 0
            } else {
                $cacheCreationMs = $prefill.elapsed_ms + $save.elapsed_ms + $restore.elapsed_ms
                $breakEvenLoadedReuses = [math]::Ceiling($cacheCreationMs / $cachedDecodeSavingMs)
            }
        }
    }

    $result = [pscustomobject]@{
        model = $modelPath
        prompt_chars = $prompt.Length
        context_size = $ContextSize
        new_tokens = $NewTokens
        cache_type = $CacheType
        prefill_gpu_layers = $PrefillGpuLayers
        decode_gpu_layers = $DecodeGpuLayers
        decode_prompt_mode = $DecodePromptMode
        used_existing_slot = $UseExistingSlot.IsPresent
        slot_file = Join-Path $slotDirPath $slotFile
        prefill_server_ready_ms = $prefillReadyMs
        prefill_elapsed_ms = if ($prefill) { $prefill.elapsed_ms } else { $null }
        prefill_tokens_evaluated = if ($prefill) { $prefill.response.tokens_evaluated } else { $null }
        prefill_tokens_cached = if ($prefill) { $prefill.response.tokens_cached } else { $null }
        save_elapsed_ms = if ($save) { $save.elapsed_ms } else { $null }
        save_n_saved = if ($save) { $save.response.n_saved } else { $null }
        save_n_written = if ($save) { $save.response.n_written } else { $null }
        decode_server_ready_ms = $decodeReadyMs
        restore_elapsed_ms = $restore.elapsed_ms
        restore_n_restored = $restore.response.n_restored
        restore_n_read = $restore.response.n_read
        phase_decode_elapsed_ms = $decode.elapsed_ms
        phase_decode_prompt_ms = $decode.response.timings.prompt_ms
        phase_decode_predicted_per_second = $decode.response.timings.predicted_per_second
        phase_decode_tokens_evaluated = $decode.response.tokens_evaluated
        phase_decode_tokens_cached = $decode.response.tokens_cached
        phase_decode_content = $decode.response.content
        phase_one_off_ms = [math]::Round($phaseOneOffMs, 3)
        cold_one_off_ms = if ($coldOneOffMs -ne $null) { [math]::Round($coldOneOffMs, 3) } else { $null }
        phase_one_off_delta_ms = if ($coldOneOffMs -ne $null) { [math]::Round($phaseOneOffMs - $coldOneOffMs, 3) } else { $null }
        cached_decode_saving_ms = if ($cachedDecodeSavingMs -ne $null) { [math]::Round($cachedDecodeSavingMs, 3) } else { $null }
        break_even_loaded_reuses = $breakEvenLoadedReuses
        cold = if ($cold) {
            [pscustomobject]@{
                server_ready_ms = $coldReadyMs
                elapsed_ms = $cold.elapsed_ms
                prompt_ms = $cold.response.timings.prompt_ms
                predicted_per_second = $cold.response.timings.predicted_per_second
                tokens_evaluated = $cold.response.tokens_evaluated
                tokens_cached = $cold.response.tokens_cached
                content = $cold.response.content
            }
        } else {
            $null
        }
    }

    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resultPath

    "# Phase-Split llama.cpp Result" | Set-Content -LiteralPath $summaryPath
    "" | Add-Content -LiteralPath $summaryPath
    "Model: ``$modelPath``" | Add-Content -LiteralPath $summaryPath
    "Prefill ``-ngl``: $PrefillGpuLayers" | Add-Content -LiteralPath $summaryPath
    "Decode ``-ngl``: $DecodeGpuLayers" | Add-Content -LiteralPath $summaryPath
    "Decode prompt mode: $DecodePromptMode" | Add-Content -LiteralPath $summaryPath
    "Cache type: $CacheType" | Add-Content -LiteralPath $summaryPath
    "Used existing slot: $($UseExistingSlot.IsPresent)" | Add-Content -LiteralPath $summaryPath
    "Prompt chars: $($prompt.Length)" | Add-Content -LiteralPath $summaryPath
    "" | Add-Content -LiteralPath $summaryPath
    if (-not $UseExistingSlot) {
        "Prefill tokens evaluated: $($result.prefill_tokens_evaluated)" | Add-Content -LiteralPath $summaryPath
        "Saved slot tokens: $($result.save_n_saved)" | Add-Content -LiteralPath $summaryPath
    }
    "Restored slot tokens: $($result.restore_n_restored)" | Add-Content -LiteralPath $summaryPath
    "Phase decode prompt ms: $($result.phase_decode_prompt_ms)" | Add-Content -LiteralPath $summaryPath
    "Phase decode tok/s: $($result.phase_decode_predicted_per_second)" | Add-Content -LiteralPath $summaryPath
    "Phase one-off ms: $($result.phase_one_off_ms)" | Add-Content -LiteralPath $summaryPath
    if ($cold) {
        "Cold decode prompt ms: $($result.cold.prompt_ms)" | Add-Content -LiteralPath $summaryPath
        "Cold decode tok/s: $($result.cold.predicted_per_second)" | Add-Content -LiteralPath $summaryPath
        "Cold one-off ms: $($result.cold_one_off_ms)" | Add-Content -LiteralPath $summaryPath
        "One-off delta ms: $($result.phase_one_off_delta_ms)" | Add-Content -LiteralPath $summaryPath
        "Cached decode saving ms: $($result.cached_decode_saving_ms)" | Add-Content -LiteralPath $summaryPath
        "Break-even loaded reuses: $($result.break_even_loaded_reuses)" | Add-Content -LiteralPath $summaryPath
    }

    Write-Host "Wrote $resultPath"
    Write-Host "Wrote $summaryPath"
    $result | ConvertTo-Json -Depth 12
} finally {
    Stop-ServerProcess $prefillServer.process
    Stop-ServerProcess $decodeServer.process
    Stop-ServerProcess $coldServer.process
}
