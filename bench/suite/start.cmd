@echo off
rem fyx benchmark suite -- Windows entry point.
rem   start.cmd                   full profile with the primary compiler, then a quick
rem                               pass with each other detected compiler (g++, clang++, cl)
rem   start.cmd --profile=quick   faster smoke run
rem   set FYX_CXX=cl              choose the primary compiler
rem   set FYX_ALL_COMPILERS=0     only the primary compiler
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%CD%"
chcp 65001 >nul

echo == fyx 基准测评：依赖检测 ==
rem ---- Visual Studio environment (needed for cl; harmless otherwise) ----
where cl >nul 2>&1
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
    if defined VSINSTALL if exist "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat" call "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)
set "COMPILERS="
for %%c in (%FYX_CXX% g++ clang++ cl) do call :addc %%c
goto :afterdetect
:addc
where %1 >nul 2>&1 || exit /b 0
for %%x in (%COMPILERS%) do if /i "%%x"=="%1" exit /b 0
set "COMPILERS=%COMPILERS% %1"
exit /b 0
:afterdetect
if "%COMPILERS%"=="" (
  echo 未找到 C++17 编译器。可任选其一安装后重试：
  echo   * MSYS2 + MinGW-w64 GCC/Clang:  winget install -e --id MSYS2.MSYS2  然后在 MSYS2 中 pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-clang 并把 ucrt64\bin 加入 PATH
  echo   * Visual Studio Build Tools:    winget install -e --id Microsoft.VisualStudio.2022.BuildTools  （勾选“使用 C++ 的桌面开发”）
  echo   * LLVM:                         winget install -e --id LLVM.LLVM
  exit /b 1
)
for /f "tokens=1" %%p in ("%COMPILERS%") do set "PRIMARY=%%p"
echo   编译器:%COMPILERS%   主编译器: %PRIMARY%
where tar >nul 2>&1 && (echo   tar: 有) || (echo   tar: 无，结果需手动压缩)
where git >nul 2>&1 && (echo   git: 有) || (echo   git: 无，仅在 third_party 缺失或更新时需要)
where rustc >nul 2>&1 && (echo   rustc: 有) || (echo   rustc: 无，Rust std sort 对手将标为不可用；可装 https://rustup.rs)

if not exist third_party\LOCK goto deps
fc /b third_party\LOCK suite\deps.lock >nul 2>&1 || goto deps
goto build
:deps
where bash >nul 2>&1 && where git >nul 2>&1 && (
  echo == 获取第三方依赖 ==
  bash suite/fetch_deps.sh third_party
) || echo 警告：third_party 缺失或与 deps.lock 不一致，且没有 bash+git；缺失的对手会被标为不可用。

:build
if not exist build mkdir build
echo == 构建 runner（%PRIMARY%）==
if /i "%PRIMARY%"=="cl" (
  cl /nologo /std:c++17 /O2 /EHsc /utf-8 suite\runner\runner.cpp /Fobuild\ /Febuild\runner.exe > build\runner_build.log 2>&1
) else (
  %PRIMARY% -std=c++17 -O2 suite/runner/runner.cpp -o build/runner.exe > build\runner_build.log 2>&1
)
if errorlevel 1 (type build\runner_build.log & echo runner 构建失败 & exit /b 1)

set "SESSION="
set "FIRST=1"
for %%c in (%COMPILERS%) do (
  if "!FIRST!"=="1" (
    echo == 主测评：%%c ==
    build\runner.exe --root="%ROOT%" --cxx=%%c --no-pack %*
    set "FIRST=0"
    if exist results\LATEST for /f "usebackq tokens=*" %%l in ("results\LATEST") do set "SESSION=!SESSION! %%l"
  ) else if not "%FYX_ALL_COMPILERS%"=="0" (
    echo == 附加编译器快速测评：%%c ==
    build\runner.exe --root="%ROOT%" --cxx=%%c --no-pack --profile=quick --skip-tests %*
    if exist results\LATEST for /f "usebackq tokens=*" %%l in ("results\LATEST") do set "SESSION=!SESSION! %%l"
  )
)
for /f "tokens=1-3 delims=/-. " %%a in ("%date%") do set "D=%%a%%b%%c"
set "OUT=fyx_results_%D%_%RANDOM%.tar.gz"
where tar >nul 2>&1 && (
  tar -czf "%OUT%" -C results %SESSION%
  echo.
  echo ==================================================================
  echo  完成。请把这个文件发回： %ROOT%\%OUT%
  echo ==================================================================
)
endlocal
