# Task Scheduler entry point: one run at a time, with a persistent local log.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Job = Join-Path $Root ".venv\Scripts\weekly-job.exe"
$LogDir = Join-Path $Root "logs"
$LockPath = Join-Path $LogDir "weekly-job.lock"

if (-not (Test-Path $Job)) {
    throw "Missing $Job. Install the project environment first."
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("weekly-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

try {
    $Lock = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    "Another weekly job is already running: $($_.Exception.Message)" |
        Out-File -FilePath $LogPath -Encoding utf8
    exit 75
}

$ExitCode = 1
try {
    "Started $(Get-Date -Format o)" | Out-File -FilePath $LogPath -Encoding utf8
    & $Job --push-docs *>> $LogPath
    $ExitCode = $LASTEXITCODE
    "Finished $(Get-Date -Format o) with exit code $ExitCode" |
        Out-File -FilePath $LogPath -Encoding utf8 -Append
}
catch {
    "Runner failed: $($_.Exception.ToString())" |
        Out-File -FilePath $LogPath -Encoding utf8 -Append
    $ExitCode = 1
}
finally {
    $Lock.Dispose()
}

# Keep two months of run logs; data and model files have separate retention.
Get-ChildItem $LogDir -Filter "weekly-*.log" |
    Where-Object LastWriteTime -lt (Get-Date).AddDays(-60) |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $ExitCode
