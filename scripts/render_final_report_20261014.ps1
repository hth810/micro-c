param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [Parameter(Mandatory=$true)][string]$PopplerDirectory
)
$ErrorActionPreference = 'Stop'
$reportSource = (Resolve-Path -LiteralPath $Source).Path
$reportRenderDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $reportRenderDirectory) { throw 'Use a new render directory; preserve prior review evidence.' }
New-Item -ItemType Directory -Path $reportRenderDirectory | Out-Null
$reportCopy = Join-Path $reportRenderDirectory 'report.docx'
$reportPdf = Join-Path $reportRenderDirectory 'report.pdf'
Copy-Item -LiteralPath $reportSource -Destination $reportCopy
Write-Output 'Creating isolated Word export instance'
$reportWord = New-Object -ComObject Word.Application
$reportDocument = $null
if ($reportWord.Documents.Count -ne 0) { throw 'Refuse to touch an instance containing user documents.' }
try {
    # Word on this host needs its minimized layout window for PDF export.
    # Never maximize it or interact with another document.
    $reportWord.WindowState = 2
    $reportWord.Visible = $true
    $reportWord.DisplayAlerts = 0
    Write-Output 'Opening exact source copy; original remains untouched'
    $reportDocument = $reportWord.Documents.Open($reportCopy, $false, $false)
    # Export computes final pagination and PAGE fields itself. An explicit
    # Repaginate call can stall Word's hidden-window layout on this host.
    Write-Output 'Exporting fixed-format PDF with native Word pagination'
    $reportDocument.SaveAs2([string]$reportPdf, 17)
    Write-Output 'Exact source copy exported'
    $reportDocument.Close(0)
    $reportDocument = $null
}
finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0) }
    if ($reportWord.Documents.Count -eq 0) { $reportWord.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportWord) | Out-Null
}
$reportPoppler = Join-Path (Resolve-Path -LiteralPath $PopplerDirectory).Path 'pdftoppm.exe'
& $reportPoppler -r 110 -png $reportPdf (Join-Path $reportRenderDirectory 'page')
if ($LASTEXITCODE -ne 0) { throw 'PDF rasterization failed.' }
Write-Output ('Render complete: {0}' -f $reportRenderDirectory)
