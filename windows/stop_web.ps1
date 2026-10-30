Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('-m drone_flight') -and $_.Name -eq 'python.exe' } | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force
  "stopped website pid $($_.ProcessId)"
}
