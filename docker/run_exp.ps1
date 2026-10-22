# Uso: powershell -File docker/run_exp.ps1 -Scene s01 -Models yolov8n,yolov8l
# Roda o pipeline completo (sem vídeos) para cada detector e grava outputs/exp/<scene>_<modelo>.log
param([string]$Scene = "s01", [string[]]$Models = @("yolov8n"))
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force outputs\exp | Out-Null
$Models = $Models | ForEach-Object { $_ -split "," }
foreach ($m in $Models) {
    "##### $m $(Get-Date -Format HH:mm:ss)"
    docker compose run --rm -e PYTHONPATH=/workspace bench -m mtmc run -c "configs/exp_${Scene}_$m.yaml" --no-viz *> "outputs\exp\${Scene}_$m.log"
    Select-String -Path "outputs\exp\${Scene}_$m.log" -Pattern 'Traceback|Error' | Select-Object -First 3 | ForEach-Object { $_.Line }
    "##### fim $m $(Get-Date -Format HH:mm:ss)"
}
