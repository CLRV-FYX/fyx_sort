# Called by update.cmd.  Downloads the branch archive (git not required),
# replaces fyx/ and the harness, keeps third_party/ unless deps.lock changed.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$repo = if ($env:FYX_REPO) { $env:FYX_REPO } elseif (Test-Path suite\REPO) { (Get-Content suite\REPO).Trim() } else { 'CLRV-FYX/fyx_sort' }
$branch = if ($env:FYX_BRANCH) { $env:FYX_BRANCH } elseif (Test-Path suite\BRANCH) { (Get-Content suite\BRANCH).Trim() } else { 'arena/01a0f1e6-fyx-sort' }
Write-Host "== 更新：$repo @ $branch =="
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("fyx_upd_" + [guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
try {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone -q --depth 1 --branch $branch "https://github.com/$repo.git" "$tmp\src"
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
        $src = "$tmp\src"
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing "https://codeload.github.com/$repo/zip/refs/heads/$branch" -OutFile "$tmp\src.zip"
        Expand-Archive "$tmp\src.zip" "$tmp\x"
        $src = (Get-ChildItem "$tmp\x" -Directory | Select-Object -First 1).FullName
    }
    if (!(Test-Path "$src\fyx_sort.hpp") -or !(Test-Path "$src\bench\suite")) { throw "下载内容不完整" }
    if (Test-Path fyx) { Remove-Item -Recurse -Force fyx }
    New-Item -ItemType Directory fyx | Out-Null
    Copy-Item "$src\fyx_sort.hpp" fyx\
    Copy-Item -Recurse "$src\test" fyx\
    foreach ($f in 'LICENSE', 'README.md') { if (Test-Path "$src\$f") { Copy-Item "$src\$f" fyx\ } }
    foreach ($d in 'core', 'algos', 'runner') {
        if (Test-Path "suite\$d") { Remove-Item -Recurse -Force "suite\$d" }
        Copy-Item -Recurse "$src\bench\suite\$d" suite\
    }
    foreach ($f in 'deps.lock', 'fetch_deps.sh', 'update.ps1') { Copy-Item "$src\bench\suite\$f" suite\ -Force }
    foreach ($f in 'start.sh', 'start.cmd', 'update.sh', 'update.cmd', 'README.md') { Copy-Item "$src\bench\suite\$f" . -Force }
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $lockSame = (Test-Path third_party\LOCK) -and ((Get-FileHash third_party\LOCK).Hash -eq (Get-FileHash suite\deps.lock).Hash)
    if (-not $lockSame) {
        if ((Get-Command bash -ErrorAction SilentlyContinue) -and (Get-Command git -ErrorAction SilentlyContinue)) {
            bash suite/fetch_deps.sh third_party
        } else {
            Write-Warning "依赖版本有变，但缺少 bash+git；请安装 Git for Windows 后重跑 update.cmd。"
        }
    }
    Write-Host "更新完成。运行 start.cmd 开始测评。"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
