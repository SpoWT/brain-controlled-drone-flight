# Installs the "DRONE APP" desktop icon and its scheduled tasks. App files: %USERPROFILE%\NeuralFlightWeb\drone_app
$app = "$env:USERPROFILE\NeuralFlightWeb\drone_app"
$py = (Get-Command python).Source
$pyw = Join-Path (Split-Path $py) "pythonw.exe"
python "$app\make_icon.py" "$app\drone.ico"
New-Item -ItemType Directory -Force "$env:USERPROFILE\Videos\Drone recordings" | Out-Null
$shell = New-Object -ComObject WScript.Shell
$path = Join-Path ([Environment]::GetFolderPath("Desktop")) "DRONE APP.lnk"
$link = $shell.CreateShortcut($path)
$link.TargetPath = $pyw
$link.Arguments = "`"$app\drone_app.pyw`""
$link.WorkingDirectory = $app
$link.IconLocation = "$app\drone.ico,0"
$link.Description = "Flying page as an app: starts everything and records the whole session"
$link.Save()
# Elevated task: restart the website and reboot the FC, no browser page.
$audit = "$env:USERPROFILE\NeuralFlightAudit"
$elevated = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$short = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
$job = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$audit\key_flight_job.ps1`" -NoPage"
Register-ScheduledTask -TaskName "NeuralFlightAppStart" -Action $job -Principal $elevated -Settings $short -Force | Out-Null
# Task that opens the app in the user's desktop session.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)
$normal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$open = New-ScheduledTaskAction -Execute $pyw -Argument "`"$app\drone_app.pyw`"" -WorkingDirectory $app
Register-ScheduledTask -TaskName "NeuralFlightDroneApp" -Action $open -Principal $normal -Settings $settings -Force | Out-Null
"installed $path ($pyw) + tasks NeuralFlightAppStart, NeuralFlightDroneApp"
