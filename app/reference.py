# In app/reference.py

import os
import subprocess
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
from typing import Dict

def ensure_soundfont(soundfont_path: str):
    """Ensure the soundfont file exists."""
    if not os.path.exists(soundfont_path):
        soundfont_dir = os.path.dirname(soundfont_path)
        os.makedirs(soundfont_dir, exist_ok=True)
        print(f"[SOUNDFONT] Note: {soundfont_path} not found. It will be downloaded on first use.")

def separate_all_stems(audio_path: str) -> Dict[str, str]:
    """
    Separate audio into stems using demucs.
    Returns a dictionary with stem names as keys and file paths as values.
    """
    try:
        print(f"[DEMUCS] Separating stems from: {audio_path}")
        
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="demucs_")
        
        # Run demucs
        cmd = f'python3 -m demucs -n htdemucs -o "{output_dir}" "{audio_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Demucs failed: {result.stderr}")
        
        # Find the output directory (demucs creates a subdirectory)
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs:
            raise Exception("Demucs did not produce output")
        
        stems_dir = os.path.join(output_dir, subdirs[0])
        
        # Map stem files
        stems = {}
        stem_files = {
            'vocals': 'vocals.wav',
            'drums': 'drums.wav',
            'bass': 'bass.wav',
            'other': 'other.wav'
        }
        
        for stem_name, stem_file in stem_files.items():
            stem_path = os.path.join(stems_dir, stem_file)
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
        
        print(f"[DEMUCS] Successfully separated stems: {list(stems.keys())}")
        return stems
    
    except Exception as e:
        print(f"[DEMUCS ERROR] {e}")
        return {}

def time_align_reference_band(band_audio: np.ndarray, band_sr: int, 
                              vocal_audio: np.ndarray, vocal_sr: int) -> np.ndarray:
    """
    Time-align the band audio to match the vocal audio length.
    This is a simple implementation that pads or trims as needed.
    """
    # Resample if necessary
    if band_sr != vocal_sr:
        band_audio = librosa.resample(band_audio, orig_sr=band_sr, target_sr=vocal_sr)
    
    # Pad or trim to match vocal length
    if len(band_audio) < len(vocal_audio):
        band_audio = np.pad(band_audio, (0, len(vocal_audio) - len(band_audio)))
    else:
        band_audio = band_audio[:len(vocal_audio)]
    
    return band_audio
