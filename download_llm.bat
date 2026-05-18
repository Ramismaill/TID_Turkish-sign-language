@echo off
REM ============================================================
REM  Qwen2.5-7B-Instruct GGUF indirme + CUDA llama-cpp-python
REM  ~4.4 GB indirme + derleme
REM ============================================================
setlocal EnableDelayedExpansion
echo.
echo ================================================
echo  LLM Kurulumu: Qwen2.5-7B-Instruct Q4_K_M
echo ================================================
echo.

call conda activate AUTSL_final
if %errorlevel% neq 0 (
    echo [HATA] AUTSL_final environment yok
    pause
    exit /b 1
)

if not exist "models\llm" mkdir models\llm
cd models\llm

echo [1/3] Qwen2.5-7B indiriliyor (~4.4 GB)...
pip install huggingface-hub --quiet
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir . --local-dir-use-symlinks False

if %errorlevel% neq 0 (
    echo [HATA] Indirme basarisiz. Manuel URL:
    echo   https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
    pause
    exit /b 1
)
cd ..\..

echo.
echo [2/3] llama-cpp-python CUDA destekli yeniden kuruluyor (10-15 dk)...
pip uninstall llama-cpp-python -y
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1
pip install llama-cpp-python==0.2.90 --no-cache-dir

if %errorlevel% neq 0 (
    echo [UYARI] CUDA surumu kurulamadi, CPU deneniyor
    pip install llama-cpp-python==0.2.90
)

echo.
echo [3/3] LLM testi
python test_llm.py

echo.
echo ================================================
echo  LLM HAZIR!
echo ================================================
pause
