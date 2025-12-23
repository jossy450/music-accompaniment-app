from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import os
from typing import Optional
import json
import numpy as np
from scipy.io import wavfile

# Import your engines
from arrangement_vocal_responsive import VocalResponsiveArrangementEngine
from audio_mixing_enhancement import AudioMixingEngine
import librosa

app = FastAPI(title="Vocal Accompaniment AI")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize engines
arrangement_engine = VocalResponsiveArrangementEngine()
mixing_engine = AudioMixingEngine(sr=44100)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/generate-with-instruments")
async def generate_with_instruments(
    vocal: UploadFile = File(...),
    backing_vocals: Optional[list[UploadFile]] = File(None),
    reference_url: Optional[str] = Form(None),
    reference_track: Optional[UploadFile] = File(None),
    selected_instruments: str = Form("drums,bass,piano"),
    style: str = Form("pop"),
    complexity: str = Form("medium"),
    mixing_profile: str = Form("vocal_forward"),
    vocal_double: float = Form(0.0),
    vocal_reverb: float = Form(0.3),
    vocal_echo: float = Form(0.0),
    vocal_harmony: float = Form(0.0),
    vocal_level: float = Form(0.0),
    music_level: float = Form(-6.0),
    apply_mastering: bool = Form(True)
):
    try:
        print(f"[API] Starting generation for style: {style}")
        
        # Save vocal file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_vocal:
            vocal_content = await vocal.read()
            tmp_vocal.write(vocal_content)
            tmp_vocal.flush()
            
            # Load and analyze vocal
            print("[API] Analyzing vocal...")
            vocal_audio, sr = librosa.load(tmp_vocal.name, sr=44100, mono=True)
            
            # Mock vocal analysis (replace with your actual analysis)
            vocal_analysis = {
                "key": "C",
                "scale": "major",
                "timing": {
                    "phrases": [
                        {"start": 0, "duration": 8, "confidence": 0.9},
                        {"start": 8, "duration": 8, "confidence": 0.9},
                        {"start": 16, "duration": 8, "confidence": 0.9},
                        {"start": 24, "duration": 8, "confidence": 0.9}
                    ],
                    "average_phrase_duration": 8.0
                },
                "intensity": {
                    "intensity": np.ones(len(vocal_audio) // 512 + 1) * 0.7
                }
            }
        
        # Load backing vocals if provided
        backing_tracks = {}
        if backing_vocals:
            for i, backing_file in enumerate(backing_vocals):
                try:
                    parts = ['treble', 'alto', 'tenor', 'bass']
                    if i < len(parts):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_backing:
                            backing_content = await backing_file.read()
                            tmp_backing.write(backing_content)
                            tmp_backing.flush()
                            
                            backing_audio, _ = librosa.load(tmp_backing.name, sr=44100, mono=True)
                            backing_tracks[parts[i]] = backing_audio
                            os.unlink(tmp_backing.name)
                except Exception as e:
                    print(f"[API] Error loading backing vocal {i}: {e}")
        
        # Generate MIDI accompaniment
        print("[API] Generating arrangement...")
        instruments = selected_instruments.split(",")
        midi = arrangement_engine.generate_vocal_responsive_arrangement(
            vocal_analysis=vocal_analysis,
            instruments=instruments,
            genre=style,
            complexity=complexity
        )
        
        # Convert MIDI to audio
        print("[API] Rendering MIDI to audio...")
        midi_audio = midi.synthesize(fs=44100)
        
        # Ensure mono
        if len(midi_audio.shape) > 1:
            midi_audio = np.mean(midi_audio, axis=1)
        
        # Apply vocal enhancement
        print("[API] Applying vocal enhancement...")
        enhanced_vocal = mixing_engine.apply_vocal_enhancement(
            audio=vocal_audio,
            reverb_amount=vocal_reverb,
            delay_amount=vocal_echo,
            double_amount=vocal_double,
            harmony_amount=vocal_harmony
        )
        
        # Mix backing vocals
        if backing_tracks:
            print("[API] Mixing backing vocals...")
            enhanced_vocal = mixing_engine.apply_backing_vocals(
                main_vocal=enhanced_vocal,
                backing_tracks=backing_tracks
            )
        
        # Mix vocal and instruments
        print("[API] Mixing vocal and instruments...")
        mixed_audio = mixing_engine.mix_vocals_with_instruments(
            vocal_audio=enhanced_vocal,
            instrument_audio=midi_audio,
            vocal_level_db=vocal_level,
            instrument_level_db=music_level
        )
        
        # Apply mastering if requested
        if apply_mastering:
            print("[API] Applying mastering...")
            mixed_audio = mixing_engine.apply_mastering(mixed_audio)
        
        # Save to temporary file
        print("[API] Saving final mix...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
            mixing_engine.save_wav(mixed_audio, tmp_output.name)
            
            # Calculate analysis data
            tempo, _ = librosa.beat.beat_track(y=mixed_audio, sr=44100)
            loudness = np.sqrt(np.mean(mixed_audio ** 2))
            
            analysis_data = {
                "key": vocal_analysis["key"],
                "tempo": float(tempo[0]) if len(tempo) > 0 else 120.0,
                "loudness": f"{20 * np.log10(loudness + 1e-10):.1f} dB",
                "duration": f"{len(mixed_audio) / 44100:.1f}s"
            }
            
            # Read the file for response
            tmp_output.flush()
            
            # Return both audio and analysis
            response_data = {
                "analysis": analysis_data,
                "audio_bytes": open(tmp_output.name, "rb").read()
            }
            
            # Clean up
            os.unlink(tmp_vocal.name)
            if os.path.exists(tmp_output.name):
                os.unlink(tmp_output.name)
            
            # Return as JSON with base64 audio
            import base64
            audio_base64 = base64.b64encode(response_data["audio_bytes"]).decode('utf-8')
            
            return {
                "status": "success",
                "analysis": response_data["analysis"],
                "audio_base64": audio_base64
            }
    
    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))