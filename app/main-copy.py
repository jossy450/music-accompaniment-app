# In app/main.py - PROFESSIONAL VOCAL-FOLLOWING VERSION

from fastapi import File, UploadFile, Form, FastAPI, Query
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

# --- Import custom modules ---
try:
    from .audio_processing import (
        professional_vocal_processing,
        enhance_vocal,
        detect_vocal_key_and_scale,
    )
    from .mixing import ProfessionalMixMaster, VocalTimeAligner
    from .arrangement_engine import VocalFollowingArrangementEngine
    from .utils import process_upload_to_wav, safe_delete_files
    from .reference import ensure_soundfont, separate_all_stems
    from .vocal_analyzer import AdvancedVocalGenreAnalyzer
    from .arrangement_selector import AdvancedArrangementSelector

    print("[IMPORT] All professional modules loaded successfully.")
except ImportError as e:
    print(f"[FATAL IMPORT ERROR] Could not import a required module: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- FastAPI app setup ---

app = FastAPI(title="Advanced Vocal Accompaniment AI - Vocal-Following Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SOUNDFONT_PATH = str(BASE_DIR / "soundfonts" / "FluidR3_GM.sf2")

# Static files + favicon
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
ensure_soundfont(SOUNDFONT_PATH)


# --- Basic endpoints ---

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(STATIC_DIR / "favicon.ico"))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "15.0-vocal-following"}


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


# --- Standalone vocal analysis for the UI ---

@app.post("/analyze-vocal")
async def analyze_vocal_endpoint(vocal: UploadFile = File(...)):
    """
    Analyze uploaded vocal and return AI recommendations:
    - Key / scale
    - Detected genre
    - Suggested arrangement (instruments + complexity)
    """
    print("[AI ANALYSIS] Starting comprehensive vocal analysis...")
    try:
        vocal_audio, sr = await process_upload_to_wav(vocal)

        # Core tonal info
        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_audio, sr)

        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_analysis = genre_analyzer.detect_genre_from_vocal(vocal_audio, sr)

        selector = AdvancedArrangementSelector()
        vocal_analysis_for_selector = {
            "detected_genre": genre_analysis["detected_genre"],
            "tempo": genre_analysis["feature_analysis"].get("tempo", 120.0),
            "melody_contour": {"phrases": []},
        }
        arrangement_profile = selector.select_best_arrangement(
            vocal_analysis_for_selector, vocal_audio, sr
        )

        result = {
            "detected_genre": genre_analysis["detected_genre"],
            "detected_key": detected_key,
            "detected_scale": detected_scale,
            "genre_probabilities": genre_analysis["genre_probabilities"],
            "feature_analysis": genre_analysis["feature_analysis"],
            "arrangement_recommendation": {
                "complexity": arrangement_profile["complexity"],
                "instruments": arrangement_profile["instruments"],
                "intensity": arrangement_profile["intensity"],
                "reasoning": arrangement_profile["reasoning"],
            },
            "recommendation": genre_analysis["recommendation"],
        }

        print(
            f"[AI ANALYSIS] Complete → Key: {detected_key} {detected_scale}, "
            f"Genre: {genre_analysis['detected_genre']}"
        )
        return result

    except Exception as e:
        msg = "Vocal analysis failed. The audio may be invalid or corrupted."
        print(f"[AI ANALYSIS ERROR] {msg}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": msg})


# --- Main generation pipeline ---

@app.post("/generate-with-instruments")
async def generate_with_instruments(
    vocal: UploadFile = File(...),
    reference_track: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),

    # arrangement / style
    selected_instruments: str = Form(""),
    style: str = Form("pop"),
    complexity: str = Form("medium"),

    # mixing
    mixing_profile: str = Form("vocal_forward"),
    vocal_level: float = Form(0.0),
    music_level: float = Form(-6.0),

    # vocal FX sliders (from frontend; treated as on/off)
    vocal_double: float = Form(0.0),
    vocal_reverb: float = Form(0.3),
    vocal_echo: float = Form(0.0),
    vocal_harmony: float = Form(0.0),

    apply_mastering: bool = Form(True),
):
    """
    VOCAL-FOLLOWING GENERATION PIPELINE

    1. Clean and analyze vocal (key, scale, genre).
    2. Either:
       - extract band from reference stems and time-align to vocal, or
       - generate MIDI accompaniment that follows vocal key/tempo.
    3. Enhance vocal (double, reverb, echo, harmony).
    4. Mix vocal + band with professional engine.
    5. Optional mastering to streaming-ready loudness.
    """
    print("[GENERATION] Starting VOCAL-FOLLOWING production...")

    temp_files_to_clean: List[str] = []
    try:
        # ===== STEP 1: LOAD + CLEAN VOCAL =====
        print("[VOCAL] Loading + professional pre-processing...")
        vocal_audio, sr = await process_upload_to_wav(vocal)
        vocal_cleaned, _ = professional_vocal_processing(vocal_audio, sr)

        # Force mono (safer for later alignment)
        if vocal_cleaned.ndim == 2:
            vocal_cleaned = np.mean(vocal_cleaned, axis=1)

        # Tonal info
        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_cleaned, sr)
        print(f"  ✓ Vocal key: {detected_key} {detected_scale}")

        # Genre / tempo for arrangement
        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_info = genre_analyzer.detect_genre_from_vocal(vocal_cleaned, sr)
        detected_genre = genre_info["detected_genre"]
        detected_tempo = genre_info["feature_analysis"].get("tempo", 120.0)

        print(
            f"  ✓ Genre: {detected_genre}, "
            f"Tempo (approx): {detected_tempo:.1f} BPM"
        )

        # ===== STEP 2: DETERMINE INSTRUMENTS =====
        instruments_list = [
            inst.strip()
            for inst in selected_instruments.split(",")
            if inst.strip()
        ]

        if not instruments_list and not reference_track and not reference_url:
            print("[ARRANGEMENT] No instruments selected → AI producer mode...")
            selector = AdvancedArrangementSelector()
            vocal_for_selector = {
                "detected_genre": detected_genre,
                "tempo": detected_tempo,
                "melody_contour": {"phrases": []},
            }
            arrangement_profile = selector.select_best_arrangement(
                vocal_for_selector, vocal_cleaned, sr
            )
            instruments_list = arrangement_profile["instruments"]
            complexity = arrangement_profile["complexity"]
            style = detected_genre
            print(
                f"  ✓ AI arrangement → style={style}, "
                f"complexity={complexity}, instruments={instruments_list}"
            )
        else:
            print(
                f"[ARRANGEMENT] Using user-selected instruments: {instruments_list}"
            )

        # ===== STEP 3: BUILD / EXTRACT ACCOMPANIMENT =====
        accompaniment_audio: Optional[np.ndarray] = None
        reference_path: Optional[str] = None

        # 3A. If reference track provided (file or URL), try DEMUCS stems
        if reference_track and reference_track.filename:
            print("[REFERENCE] Using uploaded reference track...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                content = await reference_track.read()
                tmp_ref.write(content)
                reference_path = tmp_ref.name
                temp_files_to_clean.append(reference_path)
        elif reference_url:
            print(f"[REFERENCE] Downloading reference from URL...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_ref:
                resp = requests.get(reference_url, timeout=15)
                resp.raise_for_status()
                tmp_ref.write(resp.content)
                reference_path = tmp_ref.name
                temp_files_to_clean.append(reference_path)

        if reference_path:
            try:
                print("[DEMUCS] Separating stems from reference...")
                stems = separate_all_stems(reference_path)
                temp_files_to_clean.extend(list(stems.values()))

                if stems:
                    print(f"  ✓ Stems found: {list(stems.keys())}")
                    # Choose which stems to use based on instruments_list
                    stem_paths: List[str] = []

                    for inst in instruments_list:
                        name = inst.lower()
                        if name in ("drums", "percussion") and "drums" in stems:
                            stem_paths.append(stems["drums"])
                        elif name in ("bass", "electric_bass") and "bass" in stems:
                            stem_paths.append(stems["bass"])
                        elif "other" in stems and name not in ("drums", "bass"):
                            stem_paths.append(stems["other"])

                    stem_paths = sorted(set(stem_paths))

                    if stem_paths:
                        print(f"  ✓ Mixing {len(stem_paths)} reference stems...")
                        audios = []
                        max_len = 0

                        for sp in stem_paths:
                            audio, stem_sr = sf.read(sp)
                            if stem_sr != sr:
                                audio = librosa.resample(
                                    audio, orig_sr=stem_sr, target_sr=sr
                                )
                            if audio.ndim == 2:
                                audio = np.mean(audio, axis=1)
                            max_len = max(max_len, len(audio))
                            audios.append(audio)

                        ref_mix = np.zeros(max_len, dtype=np.float32)
                        for a in audios:
                            ref_mix[: len(a)] += a

                        aligner = VocalTimeAligner(sr)
                        accompaniment_audio = aligner.align_accompaniment_to_vocal(
                            ref_mix, vocal_cleaned
                        )
                        print("  ✓ Reference band aligned to vocal.")
            except Exception as demucs_err:
                print(f"[DEMUCS ERROR] {demucs_err}")
                print("[FALLBACK] Will generate MIDI accompaniment instead.")
                accompaniment_audio = None

        # 3B. If no usable reference, generate MIDI that follows the vocal
        if accompaniment_audio is None:
            print("[MIDI] Generating vocal-following accompaniment...")
            vocal_analysis_for_engine = {
                "pitch": {"key": detected_key, "scale": detected_scale},
                "rhythm": {"tempo": detected_tempo},
                "phrases": [],  # you can later feed phrase data from an advanced analyzer
            }

            engine = VocalFollowingArrangementEngine(
                vocal_analysis=vocal_analysis_for_engine,
                genre=style,
                sr=sr,
            )
            midi = engine.create_vocal_following_arrangement(
                vocal=vocal_cleaned,
                instruments=instruments_list,
                complexity=complexity,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midi_f:
                midi.write(midi_f.name)
                midi_path = midi_f.name
                temp_files_to_clean.append(midi_path)

            accompaniment_path = midi_path.replace(".mid", ".wav")
            temp_files_to_clean.append(accompaniment_path)

            print("  ✓ Rendering MIDI via FluidR3...")
            cmd = (
                f'fluidsynth -ni "{SOUNDFONT_PATH}" "{midi_path}" '
                f'-F "{accompaniment_path}" -r {sr} -q'
            )
            subprocess.run(cmd, shell=True, check=True)

            accompaniment_audio, _ = sf.read(accompaniment_path)
            if accompaniment_audio.ndim == 2:
                accompaniment_audio = np.mean(accompaniment_audio, axis=1)

            print("  ✓ AI band rendered from MIDI.")

        # Safety: ensure accompaniment is 1D
        if accompaniment_audio.ndim == 2:
            accompaniment_audio = np.mean(accompaniment_audio, axis=1)

        # ===== STEP 4: VOCAL FX (DOUBLE / REVERB / ECHO / HARMONY) =====
        print("[VOCAL FX] Building vocal bus...")

        use_double = vocal_double > 0.05
        use_reverb = vocal_reverb > 0.05
        use_echo = vocal_echo > 0.05
        use_harmony = vocal_harmony > 0.05

        if any([use_double, use_reverb, use_echo, use_harmony]):
            lead_vocal = enhance_vocal(
                vocal_cleaned,
                sr,
                vocal_double=use_double,
                vocal_reverb=use_reverb,
                vocal_echo=use_echo,
                vocal_harmony=use_harmony,
            )
            print(
                f"  ✓ FX applied (double={use_double}, "
                f"reverb={use_reverb}, echo={use_echo}, harmony={use_harmony})"
            )
        else:
            lead_vocal = vocal_cleaned
            print("  ✓ No FX selected → clean vocal used.")

        # ===== STEP 5: MIXING =====
        print(f"[MIX] Mixing with profile: {mixing_profile}...")
        mixer = ProfessionalMixMaster(sr)
        mixed_audio = mixer.mix_tracks(
            lead_vocal,
            accompaniment_audio,
            vocal_level_db=vocal_level,
            music_level_db=music_level,
            mixing_profile=mixing_profile,
        )
        print("  ✓ Mix created.")

        # ===== STEP 6: MASTERING =====
        if apply_mastering:
            print("[MASTER] Applying mastering chain...")
            final_audio = mixer.master(mixed_audio, apply_stereo_widening=True)
            print("  ✓ Mastered to streaming loudness.")
        else:
            final_audio = mixed_audio
            print("  ✓ Mastering skipped by user.")

        # ===== STEP 7: EXPORT =====
        print("[EXPORT] Writing final WAV to memory...")
        out_buf = io.BytesIO()
        sf.write(out_buf, final_audio, sr, format="WAV", subtype="PCM_16")
        out_buf.seek(0)

        print("[GENERATION] ✅ COMPLETE – returning audio to client.")
        return StreamingResponse(
            out_buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=ai_song.wav"},
        )

    except Exception as e:
        msg = "Generation failed. Please check your input and try again."
        print(f"[GENERATION ERROR] {msg}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": msg})
    finally:
        safe_delete_files(temp_files_to_clean)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
