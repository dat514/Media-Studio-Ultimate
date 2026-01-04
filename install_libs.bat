@echo off
title Install Dependencies - Media Studio Ultimate 2.4
color 0A
echo ========================================
echo  Media Studio Ultimate 2.4
echo  Dependency Installer
echo ========================================
echo.
echo Installing required libraries...
echo.
pip install pywebview yt-dlp instaloader qrcode[pil] pillow rembg opencv-python-headless numpy librosa matplotlib scipy soundfile pandas mutagen
echo.
echo ========================================
echo  Installation complete!
echo ========================================
echo.
echo You can now run the application using:
echo  - Double-click run.bat
echo  - Or run: python main.py
echo.
pause