# Run once in PowerShell (as your user, not as SYSTEM) to start the Monday job.
# The desktop must be on. Docker is not required.

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\weekly-job.exe"
if (-not (Test-Path $Python)) {
    throw "Missing $Python. From the repo: .venv\Scripts\python.exe -m pip install -e ."
}

$Action = New-ScheduledTaskAction -Execute $Python -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 6:00AM
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun
Register-ScheduledTask -TaskName "streamflow-weekly" -Action $Action -Trigger $Trigger -Settings $Settings -Force
Write-Host "Registered streamflow-weekly for Mondays at 6:00 AM. The PC will wake if it is sleeping."
Write-Host "To run now: $Python"
Write-Host "To remove: Unregister-ScheduledTask -TaskName streamflow-weekly -Confirm:`$false"
