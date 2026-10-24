param(
    [int]$Camera = 0,
    [double]$AnalysisInterval = 5
)

# Run in the shared Conda environment, with native DLL paths available.
$ErrorActionPreference = 'Stop'
$environmentDirectory = Join-Path $env:USERPROFILE 'conda-envs\drone-sim'
$pythonExecutable = Join-Path $environmentDirectory 'python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'The shared drone-sim Conda environment was not found.'
}
$env:PATH = $environmentDirectory + ';' +
    (Join-Path $environmentDirectory 'Library\bin') + ';' +
    (Join-Path $environmentDirectory 'Scripts') + ';' + $env:PATH
$projectDirectory = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location -LiteralPath $projectDirectory
try {
    Write-Host 'Camera detection with DeepSeek responses printed in this terminal.'
    Write-Host 'If prompted, enter your API key below; it is hidden and is not saved.'
    Write-Host 'First use downloads YOLO/CLIP weights. In the camera: Q exits, A analyzes, S saves.'
    $providedKey = $false
    if (-not $env:DEEPSEEK_API_KEY) {
        Write-Host 'Paste your key with Ctrl+V or right-click, then press Enter.'
        $secureKey = Read-Host 'DeepSeek API key (hidden)' -AsSecureString
        $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        try {
            $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
            $providedKey = $true
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
            $secureKey.Dispose()
        }
        if (-not $env:DEEPSEEK_API_KEY.Trim()) {
            throw 'No API key entered. Run this script again and paste your key.'
        }
    }
    & $pythonExecutable -u -m drone_sim.vision.camera_detector --camera $Camera `
        --deepseek --analysis-interval $AnalysisInterval
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Camera stopped with an error. See the message above.' -ForegroundColor Red
    }
}
finally {
    if ($providedKey) {
        Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    }
    Pop-Location
}
