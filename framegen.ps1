param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$exe = Join-Path $PSScriptRoot ".venv\Scripts\framegen.exe"

if (-not (Test-Path $exe)) {
    Write-Error "Could not find framegen executable at '$exe'."
    exit 1
}

& $exe @Args
exit $LASTEXITCODE
