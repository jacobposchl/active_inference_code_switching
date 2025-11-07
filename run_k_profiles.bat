@echo off
REM Run K-Profiles Experiment
REM This will test M3 with k=1,2,3,4,5 profiles

echo ============================================================
echo K-PROFILES EXPERIMENT
echo ============================================================
echo.
echo This experiment will train M3 with different numbers of profiles
echo to determine the optimal k value.
echo.
echo Testing k = 1, 2, 3, 4, 5
echo.
echo Estimated time: 1-2 hours (depending on your machine)
echo.
echo Results will be saved to: results\k_profiles\
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at .venv\
    echo Continuing with system Python...
)

echo.
echo Starting experiment...
echo.

REM Run the experiment
python scripts\profile_number_sweep.py

echo.
echo ============================================================
echo EXPERIMENT COMPLETE
echo ============================================================
echo.
echo Check results\k_profiles\ for:
echo   - k_profiles_results.pkl (raw results)
echo   - k_profiles_comparison.png (visualizations)
echo   - k_profiles_summary.csv (summary table)
echo.

pause
