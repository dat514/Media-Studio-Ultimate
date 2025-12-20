import os
import io
import json
import base64
import time
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
from datetime import datetime

HISTORY_FILE = 'wave_auth_history.json'

def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_history(entry):
    hist = get_history()
    hist.insert(0, entry)
    if len(hist) > 100: 
        hist = hist[:100]
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(hist, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

def clear_history():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return True
    except:
        return False

def get_history_csv():
    hist = get_history()
    if not hist:
        return ""
    
    
    export_data = []
    for h in hist:
        row = {
            "Timestamp": h.get("timestamp", ""),
            "Filename": h.get("filename", ""),
            "Score": h.get("score", 0),
            "Verdict": h.get("verdict", ""),
            "Format": h.get("format", ""),
            "Sample Rate": h.get("sr", ""),
            "Duration": h.get("duration", ""),
            "Process Time": h.get("process_time", ""),
            "reasons": "; ".join(h.get("reasons", []))
        }
        export_data.append(row)
        
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False)



def smart_loader(file_path):
    try:
        info = sf.info(file_path)
        duration = info.duration
        sr = info.samplerate
        
        segment_duration = 60 
        y_parts = []
        
        offsets = [0, max(0, duration/2 - segment_duration/2), max(0, duration - segment_duration)]
        offsets = sorted(list(set(offsets)))
        
        for off in offsets:
            try:
                y_chunk, _ = librosa.load(file_path, sr=sr, offset=off, duration=segment_duration)
                y_trimmed, _ = librosa.effects.trim(y_chunk, top_db=60)
                if len(y_trimmed) > sr * 0.5:
                    y_parts.append(y_trimmed)
                elif len(y_chunk) > 0:
                    y_parts.append(y_chunk)
            except:
                pass 
            
        y_full = np.concatenate(y_parts) if y_parts else np.array([])
        
        y_display, _ = librosa.load(file_path, sr=sr, duration=180.0)
        
        return y_full, y_display, sr, info.format
    except Exception:
        return None, None, None, None

def analyze_audio_segment(y, sr):
    nyquist = sr / 2
    
    try:
        rolloff_99 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0]
        avg_rolloff_99 = np.mean(rolloff_99)
        bandwidth_ratio = avg_rolloff_99 / nyquist
    except:
        avg_rolloff_99 = 0
        bandwidth_ratio = 0

    try:
        rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        avg_rolloff_85 = np.mean(rolloff_85)
        slope_metric = (avg_rolloff_99 - avg_rolloff_85) / nyquist 
    except:
        slope_metric = 0.05 
        avg_rolloff_85 = 0
    
    try:
        rms = librosa.feature.rms(y=y)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)
        dynamic_range = np.abs(np.min(db)) if len(db) > 0 else 0
    except:
        dynamic_range = 0
    
    clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y) if len(y) > 0 else 0

    return {
        "cutoff_freq": avg_rolloff_99,
        "bandwidth_ratio": bandwidth_ratio,
        "slope_metric": slope_metric,
        "dynamic_range": dynamic_range,
        "clipping_ratio": clipping_ratio
    }

def estimate_mp3_profile(metrics):
    # Renamed concept to Bandwidth Util, keeping function name for safety or just replacing logic
    # The plan said rename, but keeping the name 'estimate_mp3_profile' helps avoid missed references if I missed any.
    # However, to be clean and follow the plan, I will rename it and update the caller.
    pass

def get_bandwidth_display(metrics):
    bw = metrics['bandwidth_ratio']
    # Format as percentage with 1 decimal place
    return f"{bw * 100:.1f}%"

def calculate_enterprise_score(metrics):
    score = 0
    reasons = []
    
    
    bw = metrics['bandwidth_ratio']
    if bw >= 0.95: score += 40; reasons.append("Full Spectrum (22kHz+)")
    elif bw >= 0.85: score += 35; reasons.append("Extended Bandwidth")
    elif bw >= 0.75: score += 28
    elif bw >= 0.60: score += 20
    elif bw >= 0.45: score += 12; reasons.append("Limited Bandwidth")
    else: score += 5; reasons.append("Low Bandwidth Extension")

    
    slope = metrics['slope_metric']
    if slope > 0.10: score += 30; reasons.append("Natural Spectral Decay")
    elif slope > 0.06: score += 22
    elif slope > 0.03: score += 14; reasons.append("Moderate Cutoff")
    else: score += 5; reasons.append("Sharp/Artificial Cutoff")

    
    dr = metrics['dynamic_range']
    if dr > 70: score += 25
    elif dr > 55: score += 18
    elif dr > 40: score += 12
    else: score += 5; reasons.append("Compressed/Low Dynamic Range")
    
    
    if metrics['clipping_ratio'] > 0.005:
        score -= 15
        reasons.append("⚠️ Signal Clipping Detected")
    
    
    if bw >= 0.90 and slope > 0.08 and dr > 65:
        score += 5
        reasons.append("✓ Studio-grade characteristics")
    
    final_score = max(0, min(100, score))
    
    
    if final_score >= 85: verdict = "True Lossless / Studio Master"
    elif final_score >= 70: verdict = "High Quality"
    elif final_score >= 50: verdict = "Medium Quality"
    elif final_score >= 35: verdict = "Upscaled / Transcoded"
    else: verdict = "Low Quality / Fake Lossless"
        
    return final_score, reasons, verdict

TEMP_DIR = "temp_wave_auth"

def clear_temp_images():
    try:
        if os.path.exists(TEMP_DIR):
            for f in os.listdir(TEMP_DIR):
                try:
                    os.remove(os.path.join(TEMP_DIR, f))
                except: pass
            return True
        return False
    except:
        return False

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        return {"success": False, "error": "File not found"}
    
    if not os.path.exists(TEMP_DIR):
        try: os.makedirs(TEMP_DIR)
        except: pass
        
    start_time = time.time()
    try:
        
        y_analysis, y_display, sr, fmt = smart_loader(file_path)
        
        if y_analysis is None or len(y_analysis) == 0:
            return {"success": False, "error": "Could not process audio (Empty or Corrupt)"}

        
        metrics = analyze_audio_segment(y_analysis, sr)
        
        
        score, reasons, verdict = calculate_enterprise_score(metrics)
        mp3_guess = get_bandwidth_display(metrics)
        
        
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_display, n_fft=2048)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {os.path.basename(file_path)}')
        plt.tight_layout()
        
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        
        try:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join([c for c in os.path.basename(file_path) if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            temp_filename = f"{ts_str}_{safe_name}.png"
            temp_path = os.path.join(TEMP_DIR, temp_filename)
            
            buf.seek(0)
            with open(temp_path, 'wb') as f:
                f.write(buf.read())
        except Exception as e:
            print(f"Temp save failed: {e}")
            temp_path = ""
            
        plt.close()
        
        proc_time = f"{time.time() - start_time:.2f}s"
        
        result = {
            "success": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(file_path),
            "duration": librosa.get_duration(y=y_display, sr=sr),
            "sr": sr,
            "channels": "Mono (Analysis)",
            "format": fmt,
            "cutoff": f"{int(metrics['cutoff_freq'])} Hz",
            "cutoff_freq_hz": int(metrics['cutoff_freq']), 
            "bitrate_est": "N/A", 
            "mp3_profile": mp3_guess,
            "verdict": verdict,
            "score": score,
            "reasons": reasons,
            "metrics": {k: float(f"{v:.4f}") for k,v in metrics.items()},
            "process_time": proc_time,
            "spectrogram": img_base64,
            "temp_image": temp_path
        }
        
        save_history({k:v for k,v in result.items() if k != 'spectrogram'})
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"System Error: {str(e)}"}
