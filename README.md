# Media Downloader 1.1
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue.svg)
![yt-dlp](https://img.shields.io/badge/powered_by-yt--dlp-brightgreen.svg)
![pywebview](https://img.shields.io/badge/GUI-pywebview-orange.svg)
![Single EXE](https://img.shields.io/badge/build-Single_Executable-9cf.svg)

**A beautiful, fast, and easy-to-use desktop downloader for YouTube, TikTok, Instagram, Facebook, Twitter/X, SoundCloud**  
Simple • Modern UI • Supports looping videos • Audio extraction (MP3/WAV) • Built with Python & pywebview

---
### Features
- Clean, modern dark interface with animated logo
- Supports **YouTube**, **TikTok**, **Instagram**, **Facebook**, **Twitter/X**, **SoundCloud** + thousands of sites via yt-dlp
- Download **video + audio** (MP4) or **audio only** (MP3 320kbps or lossless WAV)
- Choose video quality
- Unique **Loop & Concatenate** feature – download the same video multiple times concatenated (e.g., 10× loop for lo-fi, workout, or meme videos)
- Smart filename sanitization and duplicate handling
- One-click open download folder
- Fully offline after build – no internet needed for the app itself
- Built as a single executable (via PyInstaller) – no Python installation required for end users

---
### Screenshot
<img src="screenshots/screenshot.png" width="800"/>

---
### How to Update yt-dlp & Libraries (Important!)
#### For Developers / When Building Yourself
Always keep yt-dlp up-to-date:
```bash
# Update yt-dlp to the absolute latest version
pip install --upgrade yt-dlp
# Or force reinstall the newest nightly (sometimes needed for very recent fixes)
pip install --upgrade --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.zip
```
After updating, simply rebuild the executable:
> yt-dlp releases fixes for YouTube/TikTok/Instagram almost every week. Updating it will instantly restore downloading on sites that temporarily break.

---
### How to Build Yourself (Developers)
#### Requirements
- Python 3.9+
- pip

#### 1. Clone the repository
```bash
git clone https://github.com/dat514/MediaDownloader.git
cd MediaDownloader
```

#### 2. Install / Update dependencies
```bash
pip install --upgrade pywebview yt-dlp
```

#### 3. Run directly
```bash
python main.py
```
The executable will appear in the `dist/` folder.

---
### Supported Sites
Thanks to **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**, this tool supports **over 1000 websites**.  
Most popular ones:
- YouTube (including age-restricted, Shorts, Live)
- TikTok
- Instagram Reels / Posts / Stories
- Facebook videos
- Twitter / X
- SoundCloud
- Vimeo, Twitch clips, Reddit videos, and many more…  
Full list → https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md

---
### Known Limitations
- Some private Instagram/Facebook videos may require cookies (advanced feature not in GUI yet)
- Looping > 50× on very long videos may use significant disk space/time
- DRM-protected content (Netflix, Spotify, etc.) is **not** supported

---
### Credits & Thanks
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** – The best downloader engine
- **[pywebview](https://pywebview.flowrl.com)** – Lightweight GUI without Electron
- **[FFmpeg](https://ffmpeg.org)** – Audio extraction & video looping

---
### License
MIT License © 2025 dat514

---

<div align="center">
Just replace the placeholder screenshot and you’re good to go!  
⭐ Star this repo if you find it useful!
</div>
