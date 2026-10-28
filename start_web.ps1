# Start the flying website. -Radio: a port the drone just answered on (optional; the website finds the radio by itself).
param([string]$Radio = "")
if ($Radio -notmatch '^COM\d{1,3}$') { $Radio = "" }
$py = (Get-Command python).Source
$dir = Join-Path $env:USERPROFILE 'NeuralFlightWeb'
$mavlink = if ($Radio) { " --mavlink $Radio" } else { "" }
$cmd = "cmd /c `"`"$py`" -u ardupilot_assisted_web.py$mavlink --port 8083 > web.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $dir}
"website pid=$($r.ProcessId) rc=$($r.ReturnValue) radio=$Radio"
