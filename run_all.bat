@echo off
chcp 65001 >nul
echo ================================================================
echo                MetaGB 完整实验流程 - 快速启动
echo ================================================================
echo.

REM 检查是否已编译MetaGB库
if not exist "libmetagb.dll" (
    if not exist ".\libmetagb.so" (
        echo [警告] 未检测到MetaGB编译库，将尝试编译...
        echo.
        cd ABC_Project\metagb
        python build_metagb.py
        if errorlevel 1 (
            echo.
            echo [错误] MetaGB编译失败！
            echo 请参考 README_EXPERIMENTS.md 中的"常见问题"部分
            echo.
            pause
            exit /b 1
        )
        cd ..\..
        echo [成功] MetaGB库编译完成
        echo.
    )
)

REM 创建必要目录
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "plots" mkdir plots

echo 请选择要运行的实验模块:
echo   1. 模块A: 参数理解实验（糖尿病数据集）
echo   2. 模块B: 合成数据集性能测试
echo   3. 模块C: TabArena数据集批量测试
echo   4. 运行所有模块（A + B + C）
echo   5. 分析已有结果并生成图表
echo   0. 退出
echo.

set /p choice="请输入选项 (0-5): "

if "%choice%"=="1" (
    echo.
    echo 运行模块A: 参数理解实验...
    python run_experiments.py --module A
) else if "%choice%"=="2" (
    echo.
    echo 运行模块B: 合成数据集性能测试...
    python run_experiments.py --module B
) else if "%choice%"=="3" (
    echo.
    echo 运行模块C: TabArena数据集批量测试...
    python run_experiments.py --module C
) else if "%choice%"=="4" (
    echo.
    echo 运行所有模块...
    echo [注意] 这可能需要数小时时间，请确保环境稳定
    pause
    python run_experiments.py --module all
) else if "%choice%"=="5" (
    echo.
    echo 分析实验结果并生成图表...
    python analyze_results.py
) else if "%choice%"=="0" (
    echo.
    echo 退出
    exit /b 0
) else (
    echo.
    echo [错误] 无效的选项
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo [错误] 实验执行失败！
    echo 请查看日志文件了解详情
    pause
    exit /b 1
) else (
    echo.
    echo [成功] 执行完成！
    echo 详细结果请查看:
    echo   - 结果文件: results\
    echo   - 日志文件: logs\
    echo   - 图表文件: plots\
    echo.
    pause
)
