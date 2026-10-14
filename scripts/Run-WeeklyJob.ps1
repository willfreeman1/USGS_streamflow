# Task Scheduler entry point: one run at a time, with a persistent local log.
param(
    [string]$Job,
    [string[]]$ArgumentList = @("--push-docs"),
    [string]$LogDirectory,
    [string]$WorkingDirectory
)

$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 treats a native program's status text on the error
# stream as a terminating error when $ErrorActionPreference is Stop. The
# Monday job prints ordinary progress that way. Never let that abort a run.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function ConvertTo-CmdArgument([string]$Value) {
    if ($Value -notmatch '[ \t&()\[\]{}^=;!''+,`~"]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '""') + '"'
}

$Root = if ($WorkingDirectory) {
    $WorkingDirectory
} else {
    Split-Path -Parent $PSScriptRoot
}
if (-not $Job) {
    $Job = Join-Path $Root ".venv\Scripts\weekly-job.exe"
}
# powershell.exe -File can deliver -ArgumentList as one comma-joined string.
$ArgumentList = @(
    $ArgumentList |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$LogDir = if ($LogDirectory) { $LogDirectory } else { Join-Path $Root "logs" }
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
    $cmdParts = @(ConvertTo-CmdArgument $Job) + @(
        $ArgumentList | ForEach-Object { ConvertTo-CmdArgument $_ }
    )
    # cmd.exe appends both streams to the log. PowerShell never sees them.
    $cmdLine = "$($cmdParts -join ' ') >> `"$LogPath`" 2>&1"
    cmd.exe /c $cmdLine
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) {
        $ExitCode = 1
    }
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
