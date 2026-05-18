@echo off
REM ============================================================
REM  TID Final - Okul PC Kurulum Scripti (RTX 5060 Ti, 16 GB)
REM  Miniconda + PyTorch CUDA + ML paketleri
REM ============================================================
setlocal EnableDelayedExpansion
echo.
echo ================================================
echo  TID Final - Okul PC Kurulum
echo ================================================
echo.

where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] Conda bulunamadi!
    echo.
    echo ONCE Miniconda yukle:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo   "Add to PATH" secenegini ISARETLE
    echo.
    pause
    exit /b 1
)
echo [OK] Conda bulundu
conda --version
echo.

where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [UYARI] nvidia-smi yok. NVIDIA driver yuklu mu?
    pause
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
)
echo.

echo [1/5] Conda environment: AUTSL_final
call conda env list | findstr /C:"AUTSL_final" >nul
if %errorlevel% equ 0 (
    echo      Mevcut environment bulundu, atlaniyor
) else (
    call conda create -n AUTSL_final python=3.10 -y
)
echo.

echo [2/5] Environment aktive ediliyor
call conda activate AUTSL_final
echo.

echo [3/5] PyTorch + CUDA 12.1 (5-10 dakika)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo [4/5] Diger paketler
pip install -r requirements.txt
echo.

echo [5/5] Dogrulama
python verify_setup.py

echo.
echo ================================================
echo  KURULUM TAMAM!
echo ================================================
echo.
echo Sonraki adim:
echo   1. download_llm.bat
echo   2. python test_pipeline.py
echo.
pause
