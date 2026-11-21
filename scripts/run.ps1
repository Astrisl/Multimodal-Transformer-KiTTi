<#
.SYNOPSIS
    One entry point for the whole pipeline on Windows.

.EXAMPLE
    .\scripts\run.ps1 test          # 44 unit/integration tests (no KITTI needed)
    .\scripts\run.ps1 build         # KITTI frames -> object crops (data/kitti/crops)
    .\scripts\run.ps1 train         # train, best checkpoint -> runs/default/best.pth
    .\scripts\run.ps1 eval          # metrics -> runs/default/report.json
    .\scripts\run.ps1 demo          # picture of one classified frame -> runs/default/demo.png
    .\scripts\run.ps1 demo 000123   # same for a given frame
    .\scripts\run.ps1 export        # ONNX model -> runs/default/model.onnx
    .\scripts\run.ps1 experiments   # sparse LiDAR, error analysis, speed -> runs/default/experiments
    .\scripts\run.ps1 all           # build + train + eval + demo + export
    .\scripts\run.ps1 info          # Python / torch / CUDA / dataset status

    KITTI_ROOT defaults to <repo>\data\kitti (expects training\{image_2,velodyne,calib,label_2}).
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("info", "test", "build", "train", "eval", "demo", "export", "experiments", "all")]
    [string]$Command = "info",
    [Parameter(Position = 1)]
    [string]$SampleId
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Run = Join-Path $Root "runs\default"
$Ckpt = Join-Path $Run "best.pth"

if (-not (Test-Path $Py)) {
    throw "No virtual environment at .venv. Create it: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt; .venv\Scripts\pip install -e ."
}
if (-not $env:KITTI_ROOT) { $env:KITTI_ROOT = Join-Path $Root "data\kitti" }
$env:PYTHONUTF8 = "1"
Set-Location $Root

function Invoke-Py {
    # Python logs to stderr; under "Stop", Windows PowerShell 5.1 would turn every log line into a terminating error.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Py @args 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): python $args" }
}

function Assert-Checkpoint {
    if (-not (Test-Path $Ckpt)) { throw "No checkpoint at $Ckpt. Run: .\scripts\run.ps1 train" }
}

function Show-Info {
    Write-Host "KITTI_ROOT = $env:KITTI_ROOT"
    foreach ($d in "image_2", "velodyne", "calib", "label_2") {
        $p = Join-Path $env:KITTI_ROOT "training\$d"
        $n = if (Test-Path $p) { (Get-ChildItem $p -File).Count } else { 0 }
        Write-Host ("  training\{0,-9} {1,5} files" -f $d, $n)
    }
    $idx = Join-Path $env:KITTI_ROOT "crops\index.csv"
    $crops = if (Test-Path $idx) { (Get-Content $idx).Count - 1 } else { 0 }
    Write-Host "  crops: $crops objects"
    Write-Host "  checkpoint: $(if (Test-Path $Ckpt) { $Ckpt } else { 'none' })"
    Invoke-Py -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU)')"
}

switch ($Command) {
    "info"   { Show-Info }
    "test"   { Invoke-Py -m pytest }
    "build"  { Invoke-Py -m multimodal_transformer.data_pipeline.build_dataset --config dataset.yaml }
    "train"  { Invoke-Py -m multimodal_transformer.training.train }
    "eval"   { Assert-Checkpoint; Invoke-Py -m multimodal_transformer.training.evaluate --checkpoint $Ckpt --output (Join-Path $Run "report.json") }
    "demo"   {
        Assert-Checkpoint
        $demoArgs = @("--checkpoint", $Ckpt, "--save", (Join-Path $Run "demo.png"))
        if ($SampleId) { $demoArgs += @("--sample-id", $SampleId) }
        Invoke-Py -m multimodal_transformer.model.demo @demoArgs
    }
    "export" { Assert-Checkpoint; Invoke-Py -m multimodal_transformer.model.export --checkpoint $Ckpt --onnx (Join-Path $Run "model.onnx") }
    "experiments" {
        Assert-Checkpoint
        $expArgs = @("--checkpoint", $Ckpt, "--output-dir", (Join-Path $Run "experiments"))
        $onnx = Join-Path $Run "model.onnx"
        if (Test-Path $onnx) { $expArgs += @("--onnx", $onnx) }
        Invoke-Py -m multimodal_transformer.training.experiments @expArgs
    }
    "all"    {
        Invoke-Py -m multimodal_transformer.data_pipeline.build_dataset --config dataset.yaml
        Invoke-Py -m multimodal_transformer.training.train
        & $PSCommandPath eval
        & $PSCommandPath demo $SampleId
        & $PSCommandPath export
    }
}
