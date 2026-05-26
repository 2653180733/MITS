param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [string]$WorkDir = "D:\Code\PyCode\MITS\outputs\mits_pipeline",
    [int]$Limit = 100,
    [double]$Ratio = 15,
    [int]$MaxPairsPerSample = 32,
    [string]$QaFilter = "balanced",
    [int]$MaxPairsPerTask = 8,
    [string]$FeatureMode = "hybrid_meta",
    [double]$TextAlpha = 0.30,
    [double]$MetaAlpha = 0.15,
    [string]$GroupBy = "scene",
    [int]$MinPerGroup = 0,
    [int]$MaxPerGroup = 0,
    [switch]$AllowFullScan,
    [switch]$SkipFeatureExtraction
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$ScalSelectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

$IndexPath = Join-Path $WorkDir "mits_index.jsonl"
$ShareGPTPath = Join-Path $WorkDir "mits_sharegpt.json"
$FeatureDir = Join-Path $WorkDir "features"
$ScorePath = Join-Path $WorkDir "importance_scores.jsonl"
$SelectedIndexPath = Join-Path $WorkDir ("mits_selected_{0}.jsonl" -f $Ratio)
$SelectedShareGPTPath = Join-Path $WorkDir ("mits_selected_{0}_sharegpt.json" -f $Ratio)
$GroupSummaryPath = Join-Path $WorkDir ("mits_selected_{0}_group_summary.jsonl" -f $Ratio)

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$env:PYTHONPATH = (Join-Path $ProjectRoot "src") + ";" + $env:PYTHONPATH

$limitArgs = @("--limit", $Limit)
if ($Limit -eq 0) {
    $limitArgs = @("--limit", "0")
}
if ($AllowFullScan) {
    $limitArgs += "--allow-full-scan"
}

$selectionArgs = @(
    "--group-by", $GroupBy,
    "--min-per-group", $MinPerGroup,
    "--group-summary-output", $GroupSummaryPath
)
if ($MaxPerGroup -gt 0) {
    $selectionArgs += @("--max-per-group", $MaxPerGroup)
}

python (Join-Path $PSScriptRoot "build_mits_index.py") `
    --dataset-root $DatasetRoot `
    --output $IndexPath `
    @limitArgs

python (Join-Path $PSScriptRoot "convert_mits_to_sharegpt.py") `
    --dataset-root $DatasetRoot `
    --index $IndexPath `
    --output $ShareGPTPath `
    --max-pairs-per-sample $MaxPairsPerSample `
    --qa-filter $QaFilter `
    --max-pairs-per-task $MaxPairsPerTask

if (-not $SkipFeatureExtraction) {
    python (Join-Path $ScalSelectRoot "scripts\feature_extract_sft.py") `
        --model $ModelPath `
        --model-type qwen `
        --dataset $ShareGPTPath `
        --output-dir $FeatureDir `
        --feature-mode $FeatureMode `
        --text-alpha $TextAlpha `
        --meta-alpha $MetaAlpha `
        --max-samples -1 `
        --sample-batch-size 1 `
        --torch-dtype bfloat16 `
        --max-length 4096

    python (Join-Path $ScalSelectRoot "scripts\cur.py") `
        --features-dir $FeatureDir `
        --output $ScorePath `
        --sv-threshold 0.9

    python (Join-Path $PSScriptRoot "select_mits_subset.py") `
        --index $IndexPath `
        --cur-scores $ScorePath `
        --output $SelectedIndexPath `
        --ratio $Ratio `
        @selectionArgs
}
else {
    python (Join-Path $PSScriptRoot "select_mits_subset.py") `
        --index $IndexPath `
        --output $SelectedIndexPath `
        --ratio $Ratio `
        @selectionArgs
}

python (Join-Path $PSScriptRoot "convert_mits_to_sharegpt.py") `
    --dataset-root $DatasetRoot `
    --index $SelectedIndexPath `
    --output $SelectedShareGPTPath `
    --max-pairs-per-sample $MaxPairsPerSample `
    --qa-filter $QaFilter `
    --max-pairs-per-task $MaxPairsPerTask

Write-Host "Pipeline complete."
Write-Host "Index: $IndexPath"
Write-Host "ShareGPT: $ShareGPTPath"
Write-Host "Selected index: $SelectedIndexPath"
Write-Host "Selected ShareGPT: $SelectedShareGPTPath"
Write-Host "Group summary: $GroupSummaryPath"
