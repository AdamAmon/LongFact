$ErrorActionPreference = 'Stop'

function Get-AllStrings {
    param([object]$Value)

    if ($null -eq $Value) {
        return
    }

    if ($Value -is [string]) {
        $Value
        return
    }

    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($item in $Value.Values) {
            Get-AllStrings -Value $item
        }
        return
    }

    if (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string])) {
        foreach ($item in $Value) {
            Get-AllStrings -Value $item
        }
        return
    }

    foreach ($prop in $Value.PSObject.Properties) {
        Get-AllStrings -Value $prop.Value
    }
}

if ($null -ne $input -and @($input).Count -gt 0) {
    $inputText = (@($input) -join "`n")
} else {
    $inputText = [Console]::In.ReadToEnd()
}
if ([string]::IsNullOrWhiteSpace($inputText)) {
    $allow = @{
        hookSpecificOutput = @{
            hookEventName = 'Unknown'
            permissionDecision = 'allow'
            permissionDecisionReason = 'No hook input received.'
        }
    }
    $allow | ConvertTo-Json -Depth 8
    exit 0
}

try {
    $payload = $inputText | ConvertFrom-Json -Depth 64
} catch {
    $allow = @{
        hookSpecificOutput = @{
            hookEventName = 'Unknown'
            permissionDecision = 'allow'
            permissionDecisionReason = 'Unable to parse hook input, allowing by default.'
        }
    }
    $allow | ConvertTo-Json -Depth 8
    exit 0
}

$strings = @(Get-AllStrings -Value $payload)
$joined = ($strings -join "`n")
$eventName = $payload.hookEventName

$dangerPattern = '(?i)(git\s+reset\s+--hard|git\s+checkout\s+--|Remove-Item\s+.*-Recurse\s+.*-Force|rm\s+-rf|del\s+/s|format\s+c:|diskpart|rmdir\s+/s|rd\s+/s)'
if ($joined -match $dangerPattern) {
    $deny = @{
        hookSpecificOutput = @{
            hookEventName = $eventName
            permissionDecision = 'deny'
            permissionDecisionReason = 'LongFact guardrail blocked a potentially destructive command.'
        }
    }
    $deny | ConvertTo-Json -Depth 8
    exit 0
}

if ($eventName -eq 'PostToolUse' -and ($joined -match '(?i)\.py\b|apply_patch|create_file|edit_notebook_file')) {
    try {
        Push-Location 'D:/WBC/NJUniversity/LongFact'
        .\.venv\Scripts\python.exe -m compileall summarize retrieval nli correction eval data run_experiment.py | Out-Null
        Pop-Location
    } catch {
        if ((Get-Location).Path -ne 'D:\WBC\NJUniversity\LongFact') {
            Pop-Location -ErrorAction SilentlyContinue
        }
    }
}

$allow = @{
    hookSpecificOutput = @{
        hookEventName = $eventName
        permissionDecision = 'allow'
        permissionDecisionReason = 'LongFact guardrail check passed.'
    }
}
$allow | ConvertTo-Json -Depth 8
