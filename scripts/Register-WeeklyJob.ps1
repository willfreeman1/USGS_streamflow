# Run once in PowerShell (as your user, not as SYSTEM) to start the Monday job.
# The desktop must be on. Docker is not required.

$Root = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "Run-WeeklyJob.ps1"
if (-not (Test-Path $Runner)) {
    throw "Missing $Runner."
}

$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arguments -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 6:00AM
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Register-ScheduledTask -TaskName "streamflow-weekly" -Action $Action -Trigger $Trigger -Settings $Settings -Force
Write-Host "Registered streamflow-weekly for Mondays at 6:00 AM. The PC will wake if it is sleeping."
Write-Host "The task ignores overlapping runs, retries failures, and writes logs/weekly-*.log."
Write-Host "The job will commit and push docs/ after a successful dashboard update."
Write-Host "To run now: powershell -File `"$Runner`""
Write-Host "To remove: Unregister-ScheduledTask -TaskName streamflow-weekly -Confirm:`$false"
