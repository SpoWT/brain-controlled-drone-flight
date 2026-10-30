# Start the flying website. -Radio: a port the drone just answered on (optional; the website finds the radio by itself).
param([string]$Radio = "")
if ($Radio -notmatch '^COM\d{1,3}$') { $Radio = "" }
$found = Get-Command python -ErrorAction SilentlyContinue
if (-not $found) { Write-Error "python is not on PATH"; exit 1 }
$py = $found.Source
$dir = Split-Path $PSScriptRoot -Parent  # the repo folder
if (-not (Test-Path (Join-Path $dir 'drone_flight\__main__.py'))) { Write-Error "website not found in $dir"; exit 1 }
$mavlink = if ($Radio) { " --mavlink $Radio" } else { "" }
$cmd = "cmd /c `"`"$py`" -u -m drone_flight$mavlink --port 8083 > web.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $dir}
"website pid=$($r.ProcessId) rc=$($r.ReturnValue) radio=$Radio"
if ($r.ReturnValue -ne 0) { Write-Error "could not start the website (rc=$($r.ReturnValue))"; exit 1 }
