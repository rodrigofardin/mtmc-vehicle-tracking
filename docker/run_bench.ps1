# Uso: powershell -File docker/run_bench.ps1   (GPU livre: pare o mtmc-api antes)
Set-Location (Split-Path $PSScriptRoot -Parent)
Remove-Item outputs\detector_bench\results.json -ErrorAction SilentlyContinue
foreach ($m in @("yolov8n","yolov8l","yolov9e","yolov10l","yolov10x","yolov12x","rfdetr-medium","rfdetr-large","dfine-large","dfine-xlarge")) {
    "##### $m $(Get-Date -Format HH:mm:ss)"
    docker compose run --rm -e PYTHONPATH=/workspace bench -m mtmc.detector_bench_cli --only $m *> "outputs\detector_bench_$m.log"
}
"##### fim"
