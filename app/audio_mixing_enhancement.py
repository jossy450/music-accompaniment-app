import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import librosa
import io

class AudioMixingEngine:
    """Handles audio mixing, enhancement, and professional mastering"""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def mix_vocals_with_instruments(self, vocal_audio: np.ndarray, 
                                   instrument_audio: np.ndarray,
                                   vocal_level_db: float = 0,
                                   instrument_level_db: float = -6) -> np.ndarray:
        """Mix vocal and instrument tracks with proper level balancing"""
        
        # Ensure same length
        target_length = max(len(vocal_audio), len(instrument_audio))
        
        if len(vocal_audio) < target_length:
            vocal_audio = np.pad(vocal_audio, (0, target_length - len(vocal_audio)))
        if len(instrument_audio) < target_length:
            instrument_audio = np.pad(instrument_audio, (0, target_length - len(instrument_audio)))
        
        # Convert dB to linear gain
        vocal_gain = 10 ** (vocal_level_db / 20)
        instrument_gain = 10 ** (instrument_level_db / 20)
        
        # Apply gains
        vocal_audio = vocal_audio * vocal_gain
        instrument_audio = instrument_audio * instrument_gain
        
        # Mix with proper ducking (instruments slightly duck when vocals are present)
        mixed = vocal_audio + instrument_audio
        
        # Prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 0.9:
            mixed = mixed * (0.9 / peak)
        
        return mixed
    
    def apply_vocal_enhancement(self, audio: np.ndarray, 
                               reverb_amount: float = 0.3,
                               delay_amount: float = 0.2,
                               double_amount: float = 0.0,
                               harmony_amount: float = 0.0) -> np.ndarray:
        """Apply vocal enhancement effects"""
        enhanced = audio.copy()
        
        # Vocal doubling effect
        if double_amount > 0:
            doubled = np.roll(audio, int(self.sr * 0.01)) * double_amount
            enhanced = enhanced * (1 - double_amount) + doubled
        
        # Simple reverb (convolution with short impulse response)
        if reverb_amount > 0:
            reverb_time = 1.0  # seconds
            reverb_samples = int(self.sr * reverb_time)
            impulse = np.exp(-np.linspace(0, 10, reverb_samples))
            impulse = impulse / np.sum(impulse)
            reverb_signal = signal.convolve(audio, impulse, mode='same')
            enhanced = enhanced * (1 - reverb_amount) + reverb_signal * reverb_amount
        
        # Delay/echo effect
        if delay_amount > 0:
            delay_time = int(self.sr * 0.3)  # 300ms delay
            delayed = np.roll(audio, delay_time) * 0.7 * delay_amount
            delayed[:delay_time] = 0
            enhanced = enhanced + delayed
        
        return enhanced
    
    def apply_backing_vocals(self, main_vocal: np.ndarray, 
                            backing_tracks: dict) -> np.ndarray:
        """Mix backing vocal tracks with main vocal"""
        mixed_vocals = main_vocal.copy()
        
        for part, track in backing_tracks.items():
            if track is not None and len(track) > 0:
                # Align backing track length
                if len(track) < len(mixed_vocals):
                    track = np.pad(track, (0, len(mixed_vocals) - len(track)))
                else:
                    track = track[:len(mixed_vocals)]
                
                # Apply different gains based on vocal part
                if part in ['treble', 'soprano']:
                    gain = 0.7
                elif part == 'alto':
                    gain = 0.8
                elif part == 'tenor':
                    gain = 0.9
                elif part == 'bass':
                    gain = 0.6
                else:
                    gain = 0.7
                
                mixed_vocals = mixed_vocals + track * gain
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(mixed_vocals))
        if peak > 0.8:
            mixed_vocals = mixed_vocals * (0.8 / peak)
        
        return mixed_vocals
    
    def apply_mastering(self, audio: np.ndarray) -> np.ndarray:
        """Apply professional mastering chain"""
        mastered = audio.copy()
        
        # 1. Equalization
        # Boost high frequencies slightly
        sos = signal.butter(2, [4000, 12000], 'bandpass', fs=self.sr, output='sos')
        high_boost = signal.sosfilt(sos, mastered) * 0.3
        mastered = mastered + high_boost
        
        # 2. Multiband compression
        # Simple peak normalization
        peak = np.max(np.abs(mastered))
        if peak > 0:
            target_peak = 0.95
            mastered = mastered * (target_peak / peak)
        
        # 3. Limiter (soft clipping)
        threshold = 0.98
        mastered = np.tanh(mastered * (1/threshold)) * threshold
        
        # 4. Final loudness normalization (LUFS approximate)
        loudness = np.sqrt(np.mean(mastered ** 2))
        target_loudness = 0.1  # -14 LUFS approximate
        if loudness > 0:
            mastered = mastered * (target_loudness / loudness)
        
        return mastered
    
    def save_wav(self, audio: np.ndarray, output_path: str):
        """Save audio as 16-bit WAV file"""
        # Scale to 16-bit range
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, self.sr, audio_int16)
    
    def save_to_bytes(self, audio: np.ndarray) -> bytes:
        """Save audio to bytes for API response"""
        buffer = io.BytesIO()
        
        # Scale to 16-bit range
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to WAV in memory
        wavfile.write(buffer, self.sr, audio_int16)
        
        return buffer.getvalue()