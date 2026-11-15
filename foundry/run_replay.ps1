$ErrorActionPreference = 'Stop'
$cc = Get-Command clang, gcc, cc -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $cc) {
    throw 'A C11 compiler (clang, gcc, or cc) is required.'
}
& $cc.Source -std=c11 -O2 -Wall -Wextra -Werror -pedantic -I foundry foundry\replay.c -lm -o foundry\replay.exe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\foundry\replay.exe 32768 20
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
