from fastapi import File, UploadFile, Form, FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import sys
from typing import List, Optional
import numpy as np
import io
from pathlib import Path
import tempfile
import soundfile as sf
import subprocess
import traceback
import librosa

# --- Import project modules ---
try:
    from .audio_processing import (
        professional_vocal_processing,
        enhance_vocal,
        detect_vocal_key_and_scale,
    )
    from .mixing import ProfessionalMixMaster
    from .arrangement_engine import VocalFollowingArrangementEngine
    from .utils import process_upload_to_wav, safe_delete_files, extend_audio_to_duration
    from .reference import ensure_soundfont, separate_all_stems
    from .vocal_analyzer import AdvancedVocalGenreAnalyzer
    from .arrangement_selector import AdvancedArrangementSelector

    print("[IMPORT] All core modules loaded successfully.")
except ImportError as e:
    print(f"[FATAL IMPORT ERROR] Could not import a required module: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- FastAPI app setup ---

app = FastAPI(title="Vocal Accompaniment AI - Vocal-Following Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SOUNDFONT_PATH = os.environ.get(
    "SOUNDFONT_PATH", str(BASE_DIR / "app" / "soundfonts" / "GeneralUser-GS.sf2")
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
ensure_soundfont(SOUNDFONT_PATH)


# --- Basic endpoints ---

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/search-deezer")
async def search_deezer(q: str):
    DEEZER_API_URL = "https://api.deezer.com/search/track"
    try:
        response = requests.get(DEEZER_API_URL, params={"q": q, "limit": 5}, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for track in data.get("data", []):
            results.append({
                "id": track.get("id"),
                "title": track.get("title"),
                "artist": track.get("artist", {}).get("name"),
                "album": track.get("album", {}).get("title"),
                "preview": track.get("preview") # 30-second preview URL
            })
        
        return JSONResponse(content={"results": results})
    except requests.exceptions.RequestException as e:
        print(f"[DEEZER ERROR] {e}")
        return JSONResponse(status_code=500, content={"detail": f"Deezer API request failed: {e}"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(STATIC_DIR / "favicon.ico"))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "15.1-vocal-following"}


# --- Reference search (Deezer) ---

@app.get("/search-reference")
async def search_reference_endpoint(q: str = Query(..., min_length=3)):
    """Search for reference tracks using the Deezer API."""
    try:
        print(f"[DEEZER] Searching for: '{q}'")
        resp = requests.get(
            "https://api.deezer.com/search",
            params={"q": q, "limit": 10},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        results = []
        for t in data:
            if not t.get("preview"):
                continue
            results.append(
                {
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "artist": t.get("artist", {}).get("name", ""),
                    "duration": t.get("duration", 0),
                    "preview_url": t.get("preview"),
                }
            )
        return results
    except requests.RequestException as e:
        print(f"[DEEZER ERROR] {e}")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Could not connect to the reference track service. Please try again later."
            },
        )


# --- AI analysis endpoint ---

@app.post("/analyze-vocal")
async def analyze_vocal_endpoint(vocal: UploadFile = File(...)):
    """
    Analyze uploaded vocal and return:
    - key / scale
    - detected genre
    - suggested instruments + complexity
    """
    print("[ANALYSIS] Starting vocal analysis...")
    try:
        vocal_audio, sr = await process_upload_to_wav(vocal)

        # Key/scale
        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_audio, sr)

        # Genre
        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_info = genre_analyzer.detect_genre_from_vocal(vocal_audio, sr)

        selector = AdvancedArrangementSelector()
        vocal_for_selector = {
            "detected_genre": genre_info["detected_genre"],
            "tempo": genre_info["feature_analysis"].get("tempo", 120.0),
            "melody_contour": {"phrases": []},
        }
        arrangement_profile = selector.select_best_arrangement(
            vocal_for_selector, vocal_audio, sr
        )

        result = {
            "detected_key": detected_key,
            "detected_scale": detected_scale,
            "detected_genre": genre_info["detected_genre"],
            "genre_probabilities": genre_info["genre_probabilities"],
            "feature_analysis": genre_info["feature_analysis"],
            "arrangement_recommendation": arrangement_profile,
        }
        print(
            f"[ANALYSIS] Complete → {detected_key} {detected_scale}, genre={genre_info['detected_genre']}"
        )
        return result
    except Exception as e:
        msg = "Vocal analysis failed. Please try another file."
        print(f"[ANALYSIS ERROR] {msg}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": msg})


# --- Helper: simple time alignment for reference mix vs vocal ---

def _align_reference_to_vocal(ref_audio: np.ndarray, vocal_audio: np.ndarray) -> np.ndarray:
    """
    Simple cross-correlation based alignment.
    Shifts reference so its strongest correlation peak lines up with vocal.
    """
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    if vocal_audio.ndim > 1:
        vocal_audio = np.mean(vocal_audio, axis=1)

    # Limit length for correlation to save CPU
    max_samples = 60 * 44100  # 60s window
    ref_seg = ref_audio[:max_samples]
    vocal_seg = vocal_audio[:max_samples]

    # Normalize to avoid bias
    ref_norm = ref_seg / (np.max(np.abs(ref_seg)) + 1e-6)
    vocal_norm = vocal_seg / (np.max(np.abs(vocal_seg)) + 1e-6)

    corr = np.correlate(ref_norm, vocal_norm, mode="full")
    lag = np.argmax(corr) - (len(vocal_norm) - 1)

    if lag > 0:
        aligned = np.zeros_like(ref_audio)
        aligned[lag : lag + len(ref_audio) - lag] = ref_audio[: len(ref_audio) - lag]
    elif lag < 0:
        lag = -lag
        aligned = np.zeros_like(ref_audio)
        end = len(ref_audio) - lag
        aligned[:end] = ref_audio[lag : lag + end]
    else:
        aligned = ref_audio

    return aligned


# --- Main generation endpoint ---

@app.post("/generate-with-instruments")
async def generate_with_instruments(
    vocal: UploadFile = File(...),
    reference_track: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    selected_instruments: str = Form(""),
    style: str = Form("pop"),
    complexity: str = Form("medium"),
    mixing_profile: str = Form("vocal_forward"),
    vocal_level: float = Form(0.0),
    music_level: float = Form(-6.0),
    vocal_double: float = Form(0.0),
    vocal_reverb: float = Form(0.3),
    vocal_echo: float = Form(0.0),
    vocal_harmony: float = Form(0.0),
    vocal_arrangement: str = Form("single"),
    apply_mastering: bool = Form(True),
):
    """
    Full vocal-following pipeline:
      1. Clean & analyze vocal.
      2. Get band either from reference stems or MIDI arrangement that follows vocal.
      3. Enhance vocal with FX.
      4. Mix + master.
    """
    print("[GENERATION] Starting vocal-following generation...")

    temp_files: List[str] = []
    try:
        # === 1) Load + process vocal ===
        print("[VOCAL] Loading & cleaning...")
        vocal_audio, sr = await process_upload_to_wav(vocal)
        vocal_clean, _ = professional_vocal_processing(vocal_audio, sr)

        if vocal_clean.ndim == 2:
            vocal_clean = np.mean(vocal_clean, axis=1)

        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_clean, sr)

        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_info = genre_analyzer.detect_genre_from_vocal(vocal_clean, sr)
        detected_genre = genre_info["detected_genre"]
        detected_tempo = genre_info["feature_analysis"].get("tempo", 120.0)

        print(
            f"  ✓ Key={detected_key} {detected_scale}, "
            f"Genre={detected_genre}, Tempo≈{detected_tempo:.1f} BPM"
        )

        # === 2) Decide instruments ===
        instruments = [
            inst.strip()
            for inst in selected_instruments.split(",")
            if inst.strip()
        ]

        if not instruments and not reference_track and not reference_url:
            print("[ARRANGEMENT] No instruments selected → AI producer mode.")
            selector = AdvancedArrangementSelector()
            vocal_for_selector = {
                "detected_genre": detected_genre,
                "tempo": detected_tempo,
                "melody_contour": {"phrases": []},
            }
            arrangement_profile = selector.select_best_arrangement(
                vocal_for_selector, vocal_clean, sr
            )
            instruments = arrangement_profile["instruments"]
            complexity = arrangement_profile["complexity"]
            style = detected_genre
            print(
                f"  ✓ AI chose instruments={instruments}, complexity={complexity}, style={style}"
            )
        else:
            print(f"[ARRANGEMENT] Using user instruments: {instruments}")

        # === 3) Build accompaniment: reference stems OR MIDI ===
        band_audio: Optional[np.ndarray] = None
        reference_path: Optional[str] = None

        # 3A) Reference from file or URL
        if reference_track and reference_track.filename:
            print("[REFERENCE] Using uploaded reference...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await reference_track.read())
                reference_path = tmp.name
                temp_files.append(reference_path)
        elif reference_url:
            print("[REFERENCE] Downloading reference from URL...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                resp = requests.get(reference_url, timeout=15)
                resp.raise_for_status()
                tmp.write(resp.content)
                reference_path = tmp.name
                temp_files.append(reference_path)

        if reference_path:
            try:
                print("[DEMUCS] Separating stems...")
                stems = separate_all_stems(reference_path)
                temp_files.extend(list(stems.values()))

                if stems:
                    print(f"  ✓ Stems found: {list(stems.keys())}")
                    chosen_paths: List[str] = []

                    for inst in instruments:
                        name = inst.lower()
                        if name in ("drums", "percussion") and "drums" in stems:
                            chosen_paths.append(stems["drums"])
                        elif name in ("bass", "electric_bass") and "bass" in stems:
                            chosen_paths.append(stems["bass"])
                        else:
                            # Use "other" stem for harmonic instruments
                            if "other" in stems:
                                chosen_paths.append(stems["other"])

                    chosen_paths = sorted(set(chosen_paths))

                    if chosen_paths:
                        mixes = []
                        max_len = 0
                        for p in chosen_paths:
                            a, s_in = sf.read(p)
                            if s_in != sr:
                                a = librosa.resample(a, orig_sr=s_in, target_sr=sr)
                            if a.ndim == 2:
                                a = np.mean(a, axis=1)
                            max_len = max(max_len, len(a))
                            mixes.append(a)

                        ref_mix = np.zeros(max_len, dtype=np.float32)
                        for a in mixes:
                            ref_mix[: len(a)] += a

                        # The vocal_clean is the target duration
                        vocal_duration_s = len(vocal_clean) / sr
                        
                        # Align the reference mix to the vocal
                        aligned_mix = _align_reference_to_vocal(ref_mix, vocal_clean)
                        
                        # Extend the aligned mix to the full vocal duration
                        band_audio = extend_audio_to_duration(aligned_mix, sr, vocal_duration_s)
                        
                        print("  ✓ Reference band built, aligned, and extended to vocal duration.")
            except Exception as stem_err:
                print(f"[DEMUCS ERROR] {stem_err}")
                print("[FALLBACK] Will use MIDI accompaniment instead.")
                band_audio = None

        # 3B) MIDI arrangement if no valid reference band
        if band_audio is None:
            print("[MIDI] Generating vocal-following arrangement...")
            engine_analysis = {
                "pitch": {"key": detected_key, "scale": detected_scale},
                "rhythm": {"tempo": detected_tempo},
                "phrases": [],
            }
            engine = VocalFollowingArrangementEngine(
                vocal_analysis=engine_analysis,
                genre=style,
                sr=sr,
            )
            midi = engine.create_vocal_following_arrangement(
                vocal=vocal_clean, instruments=instruments, complexity=complexity
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midi_f:
                midi.write(midi_f.name)
                midi_path = midi_f.name
                temp_files.append(midi_path)

            wav_path = midi_path.replace(".mid", ".wav")
            temp_files.append(wav_path)

            print("  ✓ Rendering MIDI to audio with FluidSynth...")
            cmd = (
                f'fluidsynth -ni "{SOUNDFONT_PATH}" "{midi_path}" '
                f'-F "{wav_path}" -r {sr} -q'
            )
            subprocess.run(cmd, shell=True, check=True)

            band_audio, wav_sr = sf.read(wav_path)
            if wav_sr != sr:
                band_audio = librosa.resample(band_audio, orig_sr=wav_sr, target_sr=sr)
            if band_audio.ndim == 2:
                band_audio = np.mean(band_audio, axis=1)

            print("  ✓ Band rendered from MIDI.")

        # Ensure band is mono, same length envelope as vocal
        if band_audio.ndim == 2:
            band_audio = np.mean(band_audio, axis=1)

        max_len = max(len(vocal_clean), len(band_audio))
        v_bus = np.zeros(max_len, dtype=np.float32)
        a_bus = np.zeros(max_len, dtype=np.float32)
        v_bus[: len(vocal_clean)] = vocal_clean
        a_bus[: len(band_audio)] = band_audio

        # === 4) Vocal FX ===
        print("[VOCAL FX] Applying enhancements...")
        # New logic for vocal arrangement
        if vocal_arrangement == "full_backup":
            # Override user sliders for a full, rich backup sound
            use_double = True
            use_reverb = True
            use_echo = False # Keep echo off for clarity in backup
            use_harmony = True
            print("[VOCAL FX] Full Backup mode activated, overriding sliders.")
        else: # "single" or any other value
            use_double = vocal_double > 0.05
            use_reverb = vocal_reverb > 0.05
            use_echo = vocal_echo > 0.05
            use_harmony = vocal_harmony > 0.05
            print("[VOCAL FX] Single Voice mode activated, using slider values.")

        if any([use_double, use_reverb, use_echo, use_harmony]):
            vocal_bus = enhance_vocal(
                v_bus,
                sr,
                vocal_double=use_double,
                vocal_reverb=use_reverb,
                vocal_echo=use_echo,
                vocal_harmony=use_harmony,
            )
            print(
                f"  ✓ FX → double={use_double}, reverb={use_reverb}, "
                f"echo={use_echo}, harmony={use_harmony}"
            )
        else:
            vocal_bus = v_bus
            print("  ✓ FX skipped (all sliders ~0).")

        # === 5) MIX ===
        print(f"[MIX] Mixing with profile={mixing_profile}...")
        mixer = ProfessionalMixMaster(sr)
        mixed = mixer.mix_tracks(
            vocal=vocal_bus,
            band=a_bus,
            vocal_level_db=vocal_level,
            music_level_db=music_level,
            mixing_profile=mixing_profile,
        )
        print("  ✓ Mix complete.")

        # === 6) MASTER ===
        if apply_mastering:
            print("[MASTER] Applying mastering chain...")
            final_audio = mixer.master(mixed, apply_stereo_widening=True)
            print("  ✓ Mastering done.")
        else:
            final_audio = mixed
            print("  ✓ Mastering skipped.")

        # === 7) EXPORT ===
        print("[EXPORT] Encoding WAV to memory...")
        buf = io.BytesIO()
        sf.write(buf, final_audio, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        print("[GENERATION] ✅ Finished. Sending audio.")
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=ai_song.wav"},
        )

    except Exception as e:
        msg = "Generation failed. Please check your input files and try again."
        print(f"[GENERATION ERROR] {msg}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": msg})
    finally:
        safe_delete_files(temp_files)





if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
