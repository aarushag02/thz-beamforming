# Print poster.html to a 24 x 36 in PDF (and a PNG preview) with headless Chrome.
#   powershell -ExecutionPolicy Bypass -File poster\build_poster.ps1
$ErrorActionPreference = "Continue"   # Chrome logs harmless registry warnings to stderr
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$html = "file:///" + ((Join-Path $here "poster.html") -replace "\\", "/")
$pdf = Join-Path $here "Poster_24x36_3col.pdf"
$png = Join-Path $here "Poster_24x36_3col_preview.png"
$profileDir = Join-Path $env:TEMP "poster-chrome-profile"   # separate from the everyday Chrome profile
$flags = @("--headless=new", "--user-data-dir=$profileDir", "--disable-gpu", "--allow-file-access-from-files", "--hide-scrollbars",
           "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000")
& $chrome @flags "--no-pdf-header-footer" "--print-to-pdf=$pdf" $html 2>&1 | Out-Null
& $chrome @flags "--window-size=2304,3456" "--screenshot=$png" $html 2>&1 | Out-Null
Write-Output "pdf     -> $pdf"
Write-Output "preview -> $png"
