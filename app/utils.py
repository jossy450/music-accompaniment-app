# In app/utils.py

import numpy as np
import librosa
import soundfile as sf
from fastapi import UploadFile
import tempfile
import os
from pathlib import Path

# app/utils.py

import os
import tempfile
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa
from fastapi import UploadFile, HTTPException


async def process_upload_to_wav(
    upload: UploadFile,
    target_sr: int = 44100,
) -> Tuple[np.ndarray, int]:
    """
    Robustly load any uploaded/recorded audio and return (mono_audio, sr).

    Handles:
      - Browser recordings (audio/webm, .weba)
      - Common formats (wav, mp3, m4a, ogg, flac)
      - Stereo → mono conversion
      - Resampling to target_sr

    If decoding fails, raises a 400 error instead of silently producing garbage.
    """
    # 1) Save raw upload to a temp file
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        raw_path = f.name
        data = await upload.read()
        f.write(data)

    # Ensure we don't keep the upload file in memory
    await upload.close()

    audio = None
    sr = None

    try:
        # 2) First try to read directly with soundfile
        try:
            audio, sr = sf.read(raw_path)
        except Exception:
            # 3) Fallback: use ffmpeg to convert to wav
            wav_path = raw_path + ".wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                raw_path,
                "-ac",
                "1",                  # force mono
                "-ar",
                str(target_sr),       # resample here already
                wav_path,
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            audio, sr = sf.read(wav_path)
            os.remove(wav_path)

        if audio is None:
            raise RuntimeError("Decoded audio is None")

        # 4) Force mono if still multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 5) Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        else:
            audio = audio.astype(np.float32)

        # Basic sanity check
        if audio.size == 0:
            raise RuntimeError("Audio is empty after decoding")

        return audio, sr

    except subprocess.CalledProcessError:
        # ffmpeg failed → unsupported/corrupt file
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported or corrupted audio format. "
                "Please record or upload WAV, MP3, M4A, OGG, or WEBM audio."
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {e}",
        )
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass


def extend_audio_to_duration(audio: np.ndarray, sr: int, target_duration_s: float) -> np.ndarray:
    """
    Extends an audio array by looping its content until it reaches the target duration.
    If the audio is longer than the target, it is trimmed.
    """
    current_duration_s = len(audio) / sr
    
    if current_duration_s >= target_duration_s:
        # Trim the audio
        target_samples = int(target_duration_s * sr)
        return audio[:target_samples]
    
    # Calculate how many times the audio needs to be repeated
    repeat_count = int(np.ceil(target_duration_s / current_duration_s))
    
    # Loop the audio
    looped_audio = np.tile(audio, repeat_count)
    
    # Trim the looped audio to the exact target duration
    target_samples = int(target_duration_s * sr)
    return looped_audio[:target_samples]


def safe_delete_files(file_paths: list):
    """Safely delete a list of files."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[CLEANUP] Could not delete {path}: {e}")
