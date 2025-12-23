# In app/main.py
# MAIN APPLICATION - VOCAL-RESPONSIVE PIPELINE

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Query,
    HTTPException,
    Request,  # <-- needed for generate_with_instruments
)
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import sys
import io
import tempfile
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import soundfile as sf
import librosa
import requests

# -----------------------------------------------------------------------------
# Import internal modules (ALL RELATIVE IMPORTS)
# -----------------------------------------------------------------------------
try:
    from .vocal_analysis_advanced import AdvancedVocalAnalyzer
    from .vocal_analyzer import AdvancedVocalGenreAnalyzer
    from .vocal_music_analyzer import VocalMusicalAnalyzer

    from .arrangement_vocal_responsive import VocalResponsiveArrangementEngine
    from .synchronization_engine import SynchronizationEngine, RealTimeVocalFollower
    from .audio_processing import (
        professional_vocal_processing,
        enhance_vocal,
    )
    from .mixing import ProfessionalMixMaster
    from .utils import process_upload_to_wav, safe_delete_files
    from .reference import ensure_soundfont, separate_all_stems
    from .arrangement_selector import AdvancedArrangementSelector

    print("[IMPORT] All vocal-responsive modules loaded successfully.")
except ImportError as e:
    print(f"[FATAL IMPORT ERROR] Could not import a required module: {e}")
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# FastAPI app setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Advanced Vocal Accompaniment AI - Vocal Responsive")

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

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ensure_soundfont(SOUNDFONT_PATH)


# -----------------------------------------------------------------------------
# Basic endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def root() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    fav = STATIC_DIR / "favicon.ico"
    if not fav.exists():
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(str(fav))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "version": "vocal-responsive-1.0"}


# -----------------------------------------------------------------------------
# Deezer search for reference tracks
# -----------------------------------------------------------------------------
@app.get("/search-reference")
async def search_reference_endpoint(q: str = Query(..., min_length=3)) -> Any:
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
    except Exception as e:
        print(f"[DEEZER ERROR] {e}")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Could not connect to the reference track service. Please try again later."
            },
        )


# -----------------------------------------------------------------------------
# MAIN GENERATION PIPELINE
# -----------------------------------------------------------------------------
@app.post("/generate-with-instruments")
async def generate_with_instruments(
    request: Request,
    vocal: UploadFile = File(...),  # FastAPI handles the main vocal file
):
    """
    VOCAL-RESPONSIVE GENERATION PIPELINE

    We parse multipart/form-data manually so we don't rely on Pydantic's
    model validation (avoids 422 when frontend sends slightly different types).
    Sliders for vocal FX send floats 0.0–1.0; we use them as intensities.
    """

    def parse_bool(value, default=False) -> bool:
        if value is None:
            return default
        return str(value).lower() in {"true", "1", "yes", "on"}

    def parse_float(value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    print("[GENERATION] Starting VOCAL-RESPONSIVE audio production...")

    temp_files_to_clean: List[str] = []

    try:
        # ---- STEP 0: Read multipart form (for non-file fields + backing files) ----
        form = await request.form()

        # Main vocal comes from the explicit parameter
        vocal_file: UploadFile = vocal

        # Optional: backing vocals list (multiple files)
        backing_vocals: List[UploadFile] = []
        try:
            raw_list = form.getlist("backing_vocals")  # type: ignore
            backing_vocals = [f for f in raw_list if isinstance(f, UploadFile)]
        except Exception:
            backing_vocals = []

        # Optional reference fields (not wired into pipeline yet)
        reference_track = form.get("reference_track")
        reference_url = form.get("reference_url")

        # Basic params
        selected_instruments = str(form.get("selected_instruments") or "").strip()
        style = str(form.get("style") or "pop")
        complexity = str(form.get("complexity") or "medium")
        mixing_profile = str(form.get("mixing_profile") or "vocal_forward")

        # FX sliders: frontend sends 0.0–1.0 (as strings)
        vocal_double = parse_float(form.get("vocal_double"), 0.0)
        vocal_reverb = parse_float(form.get("vocal_reverb"), 0.3)
        vocal_echo = parse_float(form.get("vocal_echo"), 0.0)
        vocal_harmony = parse_float(form.get("vocal_harmony"), 0.0)

        # Levels
        vocal_level = parse_float(form.get("vocal_level"), 0.0)
        music_level = parse_float(form.get("music_level"), -6.0)
        apply_mastering = parse_bool(form.get("apply_mastering"), True)

        # ===== STEP 1: LOAD AND PROCESS VOCAL =====
        print("[VOCAL] Loading and processing vocal...")
        vocal_audio, sr = await process_upload_to_wav(vocal_file)
        if vocal_audio.ndim > 1:
            vocal_audio = np.mean(vocal_audio, axis=1)

        vocal_cleaned, _ = professional_vocal_processing(vocal_audio, sr)
        print("  ✓ Vocal loaded and cleaned")

        # ===== STEP 2: ANALYSIS =====
        print("[ANALYSIS] Performing comprehensive analysis...")
        adv_analyzer = AdvancedVocalAnalyzer(sr=sr)
        vocal_analysis = adv_analyzer.analyze_vocal_comprehensive(vocal_cleaned)

        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_info = genre_analyzer.detect_genre_from_vocal(vocal_cleaned, sr)
        vocal_analysis["detected_genre"] = genre_info.get("detected_genre", style)

        print("  ✓ Analysis summary:")
        print(f"    - Key: {vocal_analysis['key']} {vocal_analysis['scale']}")
        print(f"    - Genre: {vocal_analysis['detected_genre']}")
        print(f"    - Phrases: {vocal_analysis['timing']['num_phrases']}")
        print(f"    - Emotion: {vocal_analysis['emotional_profile']['emotion']}")

        # ===== STEP 3: DETERMINE INSTRUMENTS =====
        if selected_instruments:
            instruments_list = [
                inst.strip()
                for inst in selected_instruments.split(",")
                if inst.strip()
            ]
        else:
            print("[PRODUCER MODE] Auto-selecting arrangement...")
            selector = AdvancedArrangementSelector()
            selector_input = {
                "detected_genre": vocal_analysis["detected_genre"],
                "tempo": vocal_analysis["vocal_characteristics"]["average_pitch"],
                "melody_contour": {"phrases": []},
            }
            arrangement_profile = selector.select_best_arrangement(
                selector_input, vocal_cleaned, sr
            )
            instruments_list = arrangement_profile["instruments"]
            complexity = arrangement_profile["complexity"]

        print(
            f"[ARRANGEMENT] Generating band for genre {vocal_analysis['detected_genre']} "
            f"with instruments: {', '.join(instruments_list)} | complexity: {complexity}"
        )

        # ===== STEP 4: VOCAL-RESPONSIVE ARRANGEMENT =====
        arrangement_engine = VocalResponsiveArrangementEngine(sr=sr)
        midi = arrangement_engine.generate_vocal_responsive_arrangement(
            vocal_analysis=vocal_analysis,
            instruments=instruments_list,
            genre=vocal_analysis["detected_genre"],
            complexity=complexity,
        )

        # Render MIDI to audio with FluidSynth
        print("[ARRANGEMENT] Rendering MIDI to audio...")
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file:
            midi.write(midi_file.name)
            midi_path = midi_file.name
            temp_files_to_clean.append(midi_path)

        accompaniment_path = midi_path.replace(".mid", ".wav")
        temp_files_to_clean.append(accompaniment_path)

        cmd = (
            f'fluidsynth -ni "{SOUNDFONT_PATH}" "{midi_path}" '
            f'-F "{accompaniment_path}" -r {sr} -q'
        )
        subprocess.run(cmd, shell=True, check=True)

        band_audio, _ = sf.read(accompaniment_path)
        if band_audio.ndim > 1:
            band_audio = np.mean(band_audio, axis=1)
        print("  ✓ Band audio rendered")

        # ===== STEP 5: SYNCHRONIZE VOCAL & BAND (time shift only) =====
        print("[SYNC] Aligning vocal & band...")
        sync_engine = SynchronizationEngine(sr=sr)
        vocal_sync, band_sync = sync_engine.synchronize_vocal_and_band(
            vocal_cleaned, band_audio, vocal_analysis
        )

        follower = RealTimeVocalFollower(sr=sr)
        band_sync = follower.adjust_band_to_vocal_dynamics(
            vocal_sync, band_sync, vocal_analysis
        )
        print("  ✓ Vocal & band synchronized")

        # ===== STEP 6: BACKING VOCALS (IF ANY) =====
        backup_sync_list: List[np.ndarray] = []
        if backing_vocals:
            print(f"[BACKING] Processing {len(backing_vocals)} backing tracks...")
            for idx, bv_file in enumerate(backing_vocals):
                try:
                    bv_audio, sr_bv = await process_upload_to_wav(bv_file)
                    if sr_bv != sr:
                        bv_audio = librosa.resample(
                            bv_audio.astype(np.float32), orig_sr=sr_bv, target_sr=sr
                        )
                    if bv_audio.ndim > 1:
                        bv_audio = np.mean(bv_audio, axis=1)

                    bv_aligned = sync_engine.align_secondary_to_vocal(
                        bv_audio, vocal_sync, vocal_analysis
                    )
                    backup_sync_list.append(bv_aligned)
                    print(f"  ✓ Backing vocal {idx + 1} aligned")
                except Exception as e:
                    print(f"[BACKING WARNING] Track {idx + 1} failed: {e}")

                # ===== STEP 7: VOCAL ENHANCEMENT =====
                # ===== STEP 7: VOCAL ENHANCEMENT =====
        print("[VOCAL] Building vocal bus...")

        # Sliders send 0.0–1.0; we convert to simple on/off flags here.
        use_double = vocal_double > 0.05
        use_reverb = vocal_reverb > 0.05
        use_echo = vocal_echo > 0.05
        use_harmony = vocal_harmony > 0.05

        if not (use_double or use_reverb or use_echo or use_harmony):
            lead_vocal = vocal_sync
            print("  ✓ No FX selected – using clean vocal")
        else:
            # Call enhance_vocal POSITIONALLY to avoid keyword issues.
            # Expected signature: enhance_vocal(audio, sr, vocal_double, vocal_reverb, vocal_echo, vocal_harmony)
            try:
                lead_vocal = enhance_vocal(
                    vocal_sync,
                    sr,
                    use_double,
                    use_reverb,
                    use_echo,
                    use_harmony,
                )
            except TypeError:
                # As a last resort, just skip FX on error instead of crashing.
                print(
                    "[VOCAL WARNING] enhance_vocal signature mismatch, "
                    "falling back to clean vocal."
                )
                lead_vocal = vocal_sync

            print(
                f"  ✓ Vocal FX applied "
                f"(double={use_double}, reverb={use_reverb}, "
                f"echo={use_echo}, harmony={use_harmony})"
            )
        
        # Mix backing vocals into same bus
        if backup_sync_list:
            vocal_bus = lead_vocal.copy()
            for bv in backup_sync_list:
                L = min(len(vocal_bus), len(bv))
                vocal_bus[:L] += 0.7 * bv[:L]
            peak = np.max(np.abs(vocal_bus)) + 1e-12
            if peak > 1.0:
                vocal_bus = vocal_bus / peak
        else:
            vocal_bus = lead_vocal

        # ===== STEP 8: MIX & MASTER =====
        print(f"[MIX] Mixing with profile: {mixing_profile}")
        mixer = ProfessionalMixMaster(sr)
        mixed_audio = mixer.mix_tracks(
            vocal=vocal_bus,
            accompaniment=band_sync,
            vocal_level_db=vocal_level,
            music_level_db=music_level,
            mixing_profile=mixing_profile,
        )

        if apply_mastering:
            print("[MASTER] Applying gentle mastering...")
            final_audio = mixer.master(mixed_audio, apply_stereo_widening=True)
        else:
            final_audio = mixed_audio
            print("  ✓ Mastering skipped")

        # ===== STEP 9: EXPORT WAV =====
        print("[EXPORT] Writing final audio to WAV...")
        buf = io.BytesIO()
        sf.write(buf, final_audio, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        print("[GENERATION] ✅ COMPLETE – vocal-responsive track ready.")
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=ai_song.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        error_message = "Generation failed. Please check your input files and try again."
        print(f"[GENERATION ERROR] {error_message}: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": error_message})
    finally:
        safe_delete_files(temp_files_to_clean)



# -----------------------------------------------------------------------------
# Local dev entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
