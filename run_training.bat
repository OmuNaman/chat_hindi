@echo off
echo Starting nano_hindi training...
call venv\Scripts\activate.bat
python train.py --config 25m --total_tokens 500000000 --wandb
pause
