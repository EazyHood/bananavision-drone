$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$articleId = "7981547"
$articleUrl = "https://api.figshare.com/v2/articles/$articleId"
$filesUrl = "https://api.figshare.com/v2/articles/$articleId/files"

Write-Host "Checking PLOS/Figshare banana-UAV reference dataset..."
$article = Invoke-RestMethod -Uri $articleUrl
$files = Invoke-RestMethod -Uri $filesUrl

Write-Host "Title: $($article.title)"
Write-Host "DOI: $($article.doi)"
Write-Host "License: $($article.license.name)"
Write-Host "Public page: $($article.figshare_url)"
Write-Host "Embargoed: $($article.is_embargoed)"

if ($files.Count -eq 0) {
  Write-Host ""
  Write-Host "No public files are currently returned by the Figshare files API."
  Write-Host "Place any manually obtained files under datasets/raw/plos2019/ and document license/source."
  exit 0
}

New-Item -ItemType Directory -Force -Path "datasets/raw/plos2019" | Out-Null
foreach ($file in $files) {
  $target = Join-Path "datasets/raw/plos2019" $file.name
  Write-Host "Downloading $($file.name) -> $target"
  Invoke-WebRequest -Uri $file.download_url -OutFile $target
}
