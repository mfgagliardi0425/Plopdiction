# Set up Task Scheduler jobs for NBA automation (current user).
# NOTE: Times are in local system time. Ensure Windows time zone is ET.

$root = "C:\Users\mfgag\Sports Prediction Model"
$cmdOdds = Join-Path $root "scripts\cache_odds_snapshot.cmd"
$cmdInjury = Join-Path $root "scripts\send_injury_report_changes.cmd"
$cmdModelNow = Join-Path $root "scripts\run_model_now.cmd"
$cmdFirstGame = Join-Path $root "scripts\schedule_first_game_task.cmd"

$legacyTasks = @(
    "NBA Injury Report Daily",
    "NBA Injury Report Hourly",
    "NBA Daily Pipeline",
    "NBA Pregame Alert",
    "NBA-Daily-Pipeline",
    "NBA-First-Game-Alert",
    "NBA Pregame Model"
)

foreach ($task in $legacyTasks) {
    schtasks /Query /TN $task > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        schtasks /Delete /TN $task /F | Out-Null
    }
}

# Odds snapshot: 9am, repeat every 4 hours for 11 hours (9am-10pm)
schtasks /Create /F /TN "NBA Odds Snapshot" /TR "`"$cmdOdds`"" /SC DAILY /ST 09:00 /RI 240 /DU 11:00 | Out-Null

# Injury report: 9am, 5pm, 7pm
schtasks /Create /F /TN "NBA Injury Report 9am" /TR "`"$cmdInjury`"" /SC DAILY /ST 09:00 | Out-Null
schtasks /Create /F /TN "NBA Injury Report 5pm" /TR "`"$cmdInjury`"" /SC DAILY /ST 17:00 | Out-Null
schtasks /Create /F /TN "NBA Injury Report 7pm" /TR "`"$cmdInjury`"" /SC DAILY /ST 19:00 | Out-Null

# Model runs after injury reports
schtasks /Create /F /TN "NBA Model 9am" /TR "`"$cmdModelNow`"" /SC DAILY /ST 09:05 | Out-Null
schtasks /Create /F /TN "NBA Model 5pm" /TR "`"$cmdModelNow`"" /SC DAILY /ST 17:05 | Out-Null
schtasks /Create /F /TN "NBA Model 7pm" /TR "`"$cmdModelNow`"" /SC DAILY /ST 19:05 | Out-Null

# Schedule 1-hour-before-first-game run
schtasks /Create /F /TN "NBA First Game Scheduler" /TR "`"$cmdFirstGame`"" /SC DAILY /ST 09:00 | Out-Null

Write-Host "Scheduled tasks created/updated via schtasks."
