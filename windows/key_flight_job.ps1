# Restart the website and the flight controller, wait until the website answers, then open the page.
# Runs elevated (scheduled task). -NoPage: the Drone App opens the page itself.
# Probe /launcher-health, never "/": loading "/" takes control away from the open page.
param([switch]$NoPage)
$web = "$env:USERPROFILE\NeuralFlightWeb"
Set-Location $web
.\stop_web.ps1 | Out-Null
Start-Sleep 2
# Reboot the FC (disarmed only): clears "Roll/Pitch inconsistent" and latched battery failsafes.
# The radio is found by the drone's heartbeats on any port (COM numbers swap with plug order).
$reboot = python "$env:USERPROFILE\NeuralFlightAudit\reboot_check_radio.py" auto 2>&1 | Out-String
$radio = if ($reboot -match '"port": "(COM\d{1,3})"') { $Matches[1] } else { "" }  # where the drone answered
.\start_web.ps1 -Radio $radio | Out-Null
$ready = $false
foreach ($i in 1..30) {
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8083/launcher-health -TimeoutSec 2 | Out-Null; $ready = $true; break }
    catch { Start-Sleep 1 }
}
"$(Get-Date -Format o) website ready=$ready fc_reboot=$($reboot.Trim())" | Out-File "$env:USERPROFILE\NeuralFlightAudit\key_flight_last.log" -Encoding utf8
if (-not $NoPage) { schtasks /run /tn NeuralFlightOpenKeysPage | Out-Null }
