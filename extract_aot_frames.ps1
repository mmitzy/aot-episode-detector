# ---------------- CONFIG ---------------- #

# Root folder containing the raw AOT video files
$rawRoot = "C:\Users\danib\Projects\aot-episode-detector\AOT - RAW"

# Where to put all extracted frame folders
$outRoot = "C:\Users\danib\Projects\aot-episode-detector\data\frames"

# ffmpeg is on PATH
$ffmpeg = "ffmpeg"

# Which top-level folders to process
$topLevelFolders = @("OAD", "S1", "S2", "S3", "S4")

# Frames per second to extract
$fps = 2

# ---------------------------------------- #

if (!(Test-Path $outRoot)) {
    New-Item -ItemType Directory -Path $outRoot | Out-Null
}

foreach ($folder in $topLevelFolders) {

    $episodesDir = Join-Path (Join-Path $rawRoot $folder) "Episodes"

    if (!(Test-Path $episodesDir)) {
        Write-Host "Skipping $folder - Episodes/ not found at $episodesDir"
        continue
    }

    Write-Host "Processing group: $folder (Episodes: $episodesDir)"

    Get-ChildItem $episodesDir -Filter *.mkv | ForEach-Object {
        $file = $_

        # Extract something like S1E1, S2E10, S4Esp1, SoadE1, etc.
        $match = [regex]::Match($file.Name, "S[0-9A-Za-z]+E[0-9A-Za-z]+")

        if (-not $match.Success) {
            Write-Warning "Could not find episode code in '$($file.Name)'. Skipping."
            return
        }

        $episodeId = $match.Value   # e.g. S1E1, S4Esp1, SoadE2
        $outDir    = Join-Path $outRoot $episodeId

        if (!(Test-Path $outDir)) {
            New-Item -ItemType Directory -Path $outDir | Out-Null
        }

        Write-Host "  -> $($file.Name)  =>  $episodeId (output: $outDir)"

        # frame_000001.jpg, frame_000002.jpg, ...
        $outPattern = Join-Path $outDir "frame_%06d.jpg"

        # High-quality JPEG: q=1, explicit 8-bit JPEG pixel format
        & $ffmpeg -i $file.FullName -vf "fps=$fps" -q:v 1 -pix_fmt yuvj420p $outPattern
    }
}
