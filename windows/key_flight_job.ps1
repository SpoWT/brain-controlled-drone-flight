# Restart the website and the flight controller, wait until the website answers, then open the page.
# Runs elevated (scheduled task). -NoPage: the Drone App opens the page itself.
# Probe /launcher-health, never "/": loading "/" takes control away from the open page.
param([switch]$NoPage)
$web = Split-Path $PSScriptRoot -Parent  # the repo folder
Set-Location $web
& "$PSScriptRoot\stop_web.ps1" | Out-Null
Start-Sleep 2
# Reboot the FC (disarmed only): clears "Roll/Pitch inconsistent" and latched battery failsafes.
# The radio is found by the drone's heartbeats on any port (COM numbers swap with plug order).
$reboot = python "$PSScriptRoot\reboot_check_radio.py" auto 2>&1 | Out-String
$radio = if ($reboot -match '"port": "(COM\d{1,3})"') { $Matches[1] } else { "" }  # where the drone answered
$started = & "$PSScriptRoot\start_web.ps1" -Radio $radio 2>&1 | Out-String
$ready = $false
foreach ($i in 1..30) {
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8083/launcher-health -TimeoutSec 2 | Out-Null; $ready = $true; break }
    catch { Start-Sleep 1 }
}
"$(Get-Date -Format o) website ready=$ready start=$($started.Trim()) fc_reboot=$($reboot.Trim())" | Out-File "$PSScriptRoot\key_flight_last.log" -Encoding utf8
if (-not $ready) { Write-Error "website did not answer in 30 s: see $web\web.log"; exit 1 }
if (-not $NoPage) { schtasks /run /tn NeuralFlightOpenKeysPage | Out-Null }
