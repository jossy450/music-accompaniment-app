# In app/main.py - PROFESSIONAL VERSION (CORRECTED)

from fastapi import File, UploadFile, Form, FastAPI, Request, Query, HTTPException
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

# --- Import your custom modules ---
try:
    from .audio_processing import (
        professional_vocal_processing, enhance_vocal, 
        quantize_vocal_to_grid, detect_vocal_key_and_scale
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

app = FastAPI(title="Advanced Vocal Accompaniment AI - Professional Edition")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
SOUNDFONT_PATH = str(BASE_DIR / "soundfonts" / "FluidR3_GM.sf2")
ensure_soundfont(SOUNDFONT_PATH)

# --- API Endpoints ---

@app.get("/")
async def root(): 
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): 
    return FileResponse(str(BASE_DIR / "static" / "favicon.ico"))

@app.get("/health")
async def health(): 
    return {"status": "healthy", "version": "13.0-professional-mixing"}

@app.get("/search-reference")
async def search_reference_endpoint(q: str = Query(..., min_length=3)):
    """Search for reference tracks using the Deezer API."""
    try:
        print(f"[DEEZER] Searching for: '{q}'")
        resp = requests.get("https://api.deezer.com/search", params={"q": q, "limit": 10}, timeout=8)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        
        results = []
        for t in data:
            if not t.get("preview"): 
                continue
            results.append({
                "id": t.get("id"), 
                "title": t.get("title"),
                "artist": t.get("artist", {}).get("name", ""),
                "duration": t.get("duration", 0),
                "preview_url": t.get("preview")
            })
        return results
    except requests.RequestException as e:
        print(f"[DEEZER ERROR] {e}")
        return JSONResponse(
            status_code=503,
            content={"detail": "Could not connect to the reference track service. Please try again later."}
        )

@app.post("/analyze-vocal")
async def analyze_vocal_endpoint(vocal: UploadFile = File(...)):
    """Analyzes the uploaded vocal and returns AI recommendations."""
    print("[AI ANALYSIS] Starting comprehensive vocal analysis...")
    try:
        vocal_audio, sr = await process_upload_to_wav(vocal)
        
        # Detect key and scale
        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_audio, sr)
        
        genre_analyzer = AdvancedVocalGenreAnalyzer()
        genre_analysis = genre_analyzer.detect_genre_from_vocal(vocal_audio, sr)
        
        arrangement_selector = AdvancedArrangementSelector()
        vocal_analysis_for_selector = {
            'detected_genre': genre_analysis['detected_genre'],
            'tempo': genre_analysis['feature_analysis'].get('tempo', 120),
            'melody_contour': {'phrases': []}
        }
        arrangement_profile = arrangement_selector.select_best_arrangement(
            vocal_analysis_for_selector, vocal_audio, sr
        )
        
        final_recommendation = {
            "detected_genre": genre_analysis['detected_genre'],
            "detected_key": detected_key,
            "detected_scale": detected_scale,
            "genre_probabilities": genre_analysis['genre_probabilities'],
            "feature_analysis": genre_analysis['feature_analysis'],
            "arrangement_recommendation": {
                "complexity": arrangement_profile['complexity'],
                "instruments": arrangement_profile['instruments'],
                "intensity": arrangement_profile['intensity'],
                "reasoning": arrangement_profile['reasoning'],
            },
            "recommendation": genre_analysis['recommendation']
        }
        print(f"[AI ANALYSIS] Complete. Key: {detected_key}, Genre: {genre_analysis['detected_genre']}")
        return final_recommendation
    except Exception as e:
        error_message = f"AI analysis failed. The audio file might be invalid or corrupted. Please try a different file."
        print(f"[AI ANALYSIS ERROR] {error_message}: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": error_message}
        )

@app.post("/generate-with-instruments")
async def generate_with_instruments(
    vocal: UploadFile = File(...),
    reference_track: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    selected_instruments: str = Form(""),
    style: str = Form("pop"),
    complexity: str = Form("medium"),
    mixing_profile: str = Form("vocal_forward"),
    vocal_double: bool = Form(False),
    vocal_reverb: bool = Form(True),
    vocal_echo: bool = Form(False),
    vocal_harmony: bool = Form(False),
    vocal_level: float = Form(0.0),
    music_level: float = Form(-6.0),
    apply_mastering: bool = Form(True)
):
    """
    Professional generation pipeline:
    1. Analyze vocal (key, tempo, phrasing)
    2. Generate or extract accompaniment
    3. Align band to vocal timing
    4. Mix with selected profile
    5. Apply professional mastering
    """
    print(f"[GENERATION] Starting professional audio production...")
    
    temp_files_to_clean = []
    try:
        # ===== STEP 1: LOAD AND PROCESS VOCAL =====
        print("[VOCAL] Loading and processing...")
        vocal_audio, sr = await process_upload_to_wav(vocal)
        vocal_cleaned, vocal_processing_info = professional_vocal_processing(vocal_audio, sr)
        
        # Detect vocal characteristics
        detected_key, detected_scale = detect_vocal_key_and_scale(vocal_audio, sr)
        print(f"  ✓ Vocal Key: {detected_key}, Scale: {detected_scale}")
        
        # ===== STEP 2: DETERMINE INSTRUMENTS =====
        instruments_list = [inst.strip() for inst in selected_instruments.split(',') if inst.strip()]
        
        if not instruments_list and not reference_track and not reference_url:
            print("[PRODUCER MODE] AI selecting optimal arrangement...")
            selector = AdvancedArrangementSelector()
            genre_analyzer = AdvancedVocalGenreAnalyzer()
            genre_analysis = genre_analyzer.detect_genre_from_vocal(vocal_audio, sr)
            vocal_analysis_for_selector = {
                'detected_genre': genre_analysis['detected_genre'],
                'tempo': genre_analysis['feature_analysis'].get('tempo', 120),
                'melody_contour': {'phrases': []}
            }
            arrangement_profile = selector.select_best_arrangement(vocal_analysis_for_selector, vocal_audio, sr)
            instruments_list = arrangement_profile['instruments']
            style = genre_analysis['detected_genre']
            complexity = arrangement_profile['complexity']
            print(f"  ✓ AI Selected: Genre={style}, Complexity={complexity}, Instruments={instruments_list}")
        
        # ===== STEP 3: GENERATE OR EXTRACT ACCOMPANIMENT =====
        accompaniment_audio = None
        
        # Try to use reference track if provided
        reference_path = None
        if reference_track and reference_track.filename:
            print("[REFERENCE] Processing local reference file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                content = await reference_track.read()
                tmp_ref.write(content)
                reference_path = tmp_ref.name
                temp_files_to_clean.append(reference_path)
        elif reference_url:
            print(f"[REFERENCE] Downloading from URL...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_ref:
                response = requests.get(reference_url, timeout=15)
                response.raise_for_status()
                tmp_ref.write(response.content)
                reference_path = tmp_ref.name
                temp_files_to_clean.append(reference_path)
        
        if reference_path:
            try:
                print("[DEMUCS] Separating reference track stems...")
                all_stems = separate_all_stems(reference_path)
                temp_files_to_clean.extend(list(all_stems.values()))
                
                if all_stems:
                    print(f"  ✓ Successfully extracted stems: {list(all_stems.keys())}")
                    
                    # Mix selected stems
                    stems_to_mix = []
                    for instrument in instruments_list:
                        if instrument in ['drums', 'percussion'] and 'drums' in all_stems:
                            stems_to_mix.append(all_stems['drums'])
                        elif instrument in ['bass', 'electric_bass'] and 'bass' in all_stems:
                            stems_to_mix.append(all_stems['bass'])
                        elif instrument not in ['drums', 'bass'] and 'other' in all_stems:
                            stems_to_mix.append(all_stems['other'])
                    
                    stems_to_mix = sorted(list(set(stems_to_mix)))
                    
                    if stems_to_mix:
                        print(f"  ✓ Mixing {len(stems_to_mix)} stems...")
                        mixed_band_audio = None
                        max_len = 0
                        audios = []
                        
                        for stem_path in stems_to_mix:
                            audio, stem_sr = sf.read(stem_path)
                            if stem_sr != sr:
                                audio = librosa.resample(audio, orig_sr=stem_sr, target_sr=sr)
                            if len(audio) > max_len:
                                max_len = len(audio)
                            audios.append(audio)
                        
                        mixed_band_audio = np.zeros(max_len)
                        for audio in audios:
                            mixed_band_audio[:len(audio)] += audio
                        
                        # ===== STEP 4: ALIGN BAND TO VOCAL =====
                        print("[ALIGNMENT] Aligning band to vocal timing...")
                        aligner = VocalTimeAligner(sr)
                        accompaniment_audio = aligner.align_accompaniment_to_vocal(mixed_band_audio, vocal_cleaned)
                        print("  ✓ Successfully aligned reference band to vocal")
            except Exception as demucs_error:
                print(f"[DEMUCS ERROR] Failed to process reference: {demucs_error}")
                print("[FALLBACK] Switching to AI MIDI generation...")
                accompaniment_audio = None
        
        # If no valid reference accompaniment, generate with AI
        if accompaniment_audio is None:
            print("[MIDI GENERATION] Creating AI accompaniment...")
            
            # Analyze vocal phrasing
            genre_analyzer = AdvancedVocalGenreAnalyzer()
            vocal_analysis = genre_analyzer.detect_genre_from_vocal(vocal_audio, sr)
            
            # Create vocal-following arrangement
            vocal_analysis_for_engine = {
                'pitch': {'key': detected_key, 'scale': detected_scale},
                'rhythm': {'tempo': vocal_analysis['feature_analysis'].get('tempo', 120)},
                'phrases': []
            }
            
            engine = VocalFollowingArrangementEngine(vocal_analysis_for_engine, style, sr)
            midi = engine.create_vocal_following_arrangement(vocal_cleaned, instruments_list, complexity)
            
            # Render MIDI to audio
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file:
                midi.write(midi_file.name)
                midi_path = midi_file.name
                temp_files_to_clean.append(midi_path)
            
            accompaniment_path = midi_path.replace(".mid", ".wav")
            temp_files_to_clean.append(accompaniment_path)
            
            print(f"  ✓ Rendering MIDI to audio...")
            render_cmd = f'fluidsynth -ni "{SOUNDFONT_PATH}" "{midi_path}" -F "{accompaniment_path}" -r {sr} -q'
            subprocess.run(render_cmd, shell=True, check=True)
            
            # Force mono to match sync engine expectations
            accompaniment_audio, _ = sf.read(accompaniment_path)

        if  accompaniment_audio.ndim == 2:   # stereo → mono
            accompaniment_audio = np.mean(accompaniment_audio, axis=1)
            print("  ✓ AI accompaniment generated successfully")
        
        # ===== STEP 5: ENHANCE VOCAL =====
        print("[ENHANCEMENT] Applying vocal enhancements...")
        enhanced_vocal = enhance_vocal(vocal_cleaned, sr, vocal_double, vocal_reverb, 
                                       vocal_echo, vocal_harmony)
        print("  ✓ Vocal enhancements applied")
        
        # ===== STEP 6: PROFESSIONAL MIXING =====
        print(f"[MIXING] Mixing with profile: {mixing_profile}...")
        mixer = ProfessionalMixMaster(sr)
        mixed_audio = mixer.mix_tracks(enhanced_vocal, accompaniment_audio, 
                                       vocal_level_db=vocal_level, 
                                       music_level_db=music_level,
                                       mixing_profile=mixing_profile)
        print(f"  ✓ Mixed using {mixing_profile} profile")
        
        # ===== STEP 7: PROFESSIONAL MASTERING =====
        if apply_mastering:
            print("[MASTERING] Applying professional mastering chain...")
            final_audio = mixer.master(mixed_audio, apply_stereo_widening=True)
            print("  ✓ Mastering applied (Multiband Compression, EQ, Limiting, Loudness Normalization)")
        else:
            final_audio = mixed_audio
            print("  ✓ Mastering skipped (user preference)")
        
        # ===== STEP 8: EXPORT =====
        print("[EXPORT] Preparing final audio...")
        output_io = io.BytesIO()
        sf.write(output_io, final_audio, sr, format='WAV', subtype='PCM_16')
        output_io.seek(0)
        
        print("[GENERATION] ✓ Complete! Sending to user.")
        return StreamingResponse(output_io, media_type="audio/wav", 
                                headers={"Content-Disposition": "attachment; filename=ai_song.wav"})
    
    except Exception as e:
        error_message = f"Generation failed. Please check your input files and try again."
        print(f"[GENERATION ERROR] {error_message}: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": error_message}
        )
    finally:
        safe_delete_files(temp_files_to_clean)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
