# In app/synchronization_engine.py
# SYNCHRONIZATION ENGINE - Ensures perfect timing alignment between vocal and band

import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Dict


class SynchronizationEngine:
    """
    Ensures timing synchronization between vocal and generated band.

    Handles:
    - Making vocal and band the same length
    - Aligning band to vocal phrase boundaries
    - Quantizing both to a common tempo grid
    - Phase alignment
    - Aligning secondary tracks (backing vocals) to the lead
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def synchronize_vocal_and_band(
        self,
        vocal_audio: np.ndarray,
        band_audio: np.ndarray,
        vocal_analysis: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronize vocal and band so they play together tightly.

        Both inputs are treated as MONO. If they are stereo, they are converted
        to mono internally by averaging channels.

        Returns:
            (synchronized_vocal, synchronized_band)
        """
        print("[SYNC] Synchronizing vocal and band...")

        # Force mono
        vocal_audio = self._ensure_mono(vocal_audio)
        band_audio = self._ensure_mono(band_audio)

        # Step 1: Ensure both are the same length
        max_length = max(len(vocal_audio), len(band_audio))
        vocal_sync = np.zeros(max_length, dtype=np.float32)
        band_sync = np.zeros(max_length, dtype=np.float32)

        vocal_sync[: len(vocal_audio)] = vocal_audio
        band_sync[: len(band_audio)] = band_audio

        # Step 2: Align band to vocal phrase boundaries (light energy-based fix)
        phrases = vocal_analysis.get("timing", {}).get("phrases", [])
        if phrases:
            band_sync = self._align_to_phrase_boundaries(band_sync, phrases)

        # Step 3: Quantize both to a common tempo grid
        tempo = self._estimate_tempo(vocal_analysis)
        vocal_sync, band_sync = self._quantize_to_grid(vocal_sync, band_sync, tempo)

        # Step 4: Phase alignment by cross-correlation
        vocal_sync, band_sync = self._align_phase(vocal_sync, band_sync)

        print(
            f"  ✓ Synchronization complete. Duration: {len(vocal_sync) / self.sr:.2f}s"
        )

        return vocal_sync, band_sync

    def align_secondary_to_vocal(
        self,
        secondary_audio: np.ndarray,
        lead_vocal: np.ndarray,
        vocal_analysis: Dict | None = None,
    ) -> np.ndarray:
        """
        Align a secondary track (e.g., backing vocal) to the already-synced lead.

        - Forces mono
        - Matches length to lead
        - Uses cross-correlation to find best lag
        - Optionally quantizes to the same tempo grid as the lead

        Returns:
            aligned_secondary (same length as lead_vocal)
        """
        print("[SYNC] Aligning secondary track to lead vocal...")

        secondary = self._ensure_mono(secondary_audio)
        lead = self._ensure_mono(lead_vocal)

        target_len = len(lead)

        # Basic pad/trim to approximate length
        if len(secondary) < target_len:
            pad = target_len - len(secondary)
            secondary = np.pad(secondary, (0, pad), mode="constant")
        else:
            secondary = secondary[:target_len]

        # Cross-correlation for best lag
        corr = signal.correlate(secondary, lead, mode="full")
        lag = np.argmax(corr) - (len(lead) - 1)

        if lag > 0:
            # secondary starts later than lead → shift left
            secondary = np.pad(secondary[lag:], (0, lag), mode="constant")
        elif lag < 0:
            # secondary starts earlier → shift right
            lag_abs = -lag
            secondary = np.pad(secondary, (lag_abs, 0), mode="constant")[:target_len]

        secondary = secondary[:target_len]

        # Optional: quantize to tempo grid using same tempo as lead/band
        if vocal_analysis is not None:
            tempo = self._estimate_tempo(vocal_analysis)
            beat_samples = int((60.0 / tempo) * self.sr)
            secondary = self._quantize_audio(secondary, beat_samples)

        print("  ✓ Secondary track aligned to lead")
        return secondary

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo/multi-channel to mono if needed."""
        if audio.ndim > 1:
            return np.mean(audio, axis=1).astype(np.float32)
        return audio.astype(np.float32)

    def _align_to_phrase_boundaries(
        self, band_audio: np.ndarray, phrases: list
    ) -> np.ndarray:
        """Basic energy normalization within vocal phrases to avoid clipping."""
        print("  [PHRASE ALIGNMENT] Adjusting band energy inside vocal phrases...")

        for phrase in phrases:
            start_sample = int(phrase["start"] * self.sr)
            end_sample = int(phrase["end"] * self.sr)

            if start_sample >= len(band_audio):
                continue
            end_sample = min(end_sample, len(band_audio))
            if end_sample <= start_sample:
                continue

            region = band_audio[start_sample:end_sample]
            max_abs = np.max(np.abs(region)) if region.size > 0 else 0.0
            if max_abs > 0.95:
                band_audio[start_sample:end_sample] *= 0.95 / max_abs

        return band_audio

    def _quantize_to_grid(
        self, vocal_audio: np.ndarray, band_audio: np.ndarray, tempo: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize both vocal and band to a common tempo grid."""
        print(f"  [QUANTIZATION] Quantizing to {tempo:.0f} BPM grid...")

        beat_samples = int((60.0 / tempo) * self.sr)
        vocal_q = self._quantize_audio(vocal_audio, beat_samples)
        band_q = self._quantize_audio(band_audio, beat_samples)
        return vocal_q, band_q

    def _quantize_audio(self, audio: np.ndarray, grid_size: int) -> np.ndarray:
        """
        Quantize audio by gently time-stretching around onsets to align them
        with the nearest grid points. This is deliberately conservative to
        avoid artifacts.
        """
        audio = self._ensure_mono(audio)
        quantized = np.copy(audio)

        # Onset detection
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        onset_frames = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10
        )
        onset_samples = librosa.frames_to_samples(
            onset_frames, hop_length=self.hop_length
        )

        for onset in onset_samples:
            grid_pos = int(np.round(onset / grid_size) * grid_size)
            if grid_pos < 0 or grid_pos >= len(quantized):
                continue

            shift = grid_pos - onset
            # Only do micro-adjustments
            if abs(shift) < grid_size // 4 and len(quantized) > grid_size:
                # Very light global time-stretch; you may want something more advanced later
                factor = 1.0 + (shift / grid_size) * 0.02
                quantized = librosa.effects.time_stretch(quantized, rate=factor)
                quantized = self._ensure_mono(quantized)

        # Ensure same length as original
        if len(quantized) > len(audio):
            quantized = quantized[: len(audio)]
        elif len(quantized) < len(audio):
            pad = len(audio) - len(quantized)
            quantized = np.pad(quantized, (0, pad))

        return quantized

    def _align_phase(
        self, vocal_audio: np.ndarray, band_audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align phase between vocal and band using cross-correlation."""
        print("  [PHASE ALIGNMENT] Aligning phase between vocal and band...")

        vocal_audio = self._ensure_mono(vocal_audio)
        band_audio = self._ensure_mono(band_audio)

        min_len = min(len(vocal_audio), len(band_audio))
        vocal_audio = vocal_audio[:min_len]
        band_audio = band_audio[:min_len]

        # Cross-correlation, center lag at 0
        correlation = signal.correlate(band_audio, vocal_audio, mode="full")
        lag = np.argmax(correlation) - (len(vocal_audio) - 1)

        if lag > 0:
            # band starts later → shift left
            band_audio = np.pad(band_audio[lag:], (0, lag), mode="constant")
        elif lag < 0:
            # band starts earlier → shift right
            lag_abs = -lag
            band_audio = np.pad(band_audio, (lag_abs, 0), mode="constant")[:min_len]

        return vocal_audio, band_audio

    def _estimate_tempo(self, vocal_analysis: Dict) -> float:
        """Estimate tempo from vocal analysis."""
        timing = vocal_analysis.get("timing", {})
        phrases = timing.get("phrases", [])

        if not phrases:
            return 120.0

        avg_phrase_duration = timing.get("average_phrase_duration", 3.0)

        if avg_phrase_duration < 2:
            return 140.0
        elif avg_phrase_duration < 3:
            return 120.0
        elif avg_phrase_duration < 5:
            return 100.0
        else:
            return 80.0


class RealTimeVocalFollower:
    """
    Follows vocal in real-time, adjusting band parameters dynamically.
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512

    def adjust_band_to_vocal_dynamics(
        self,
        vocal_audio: np.ndarray,
        band_audio: np.ndarray,
        vocal_analysis: Dict,
    ) -> np.ndarray:
        """
        Adjust band dynamics to match vocal intensity curve.
        """
        print("[DYNAMICS] Adjusting band dynamics to vocal intensity...")

        vocal_intensity = vocal_analysis["intensity"]["intensity"]

        band_audio = band_audio.astype(np.float32)
        band_rms = librosa.feature.rms(y=band_audio, hop_length=self.hop_length)[0]
        band_rms_norm = (band_rms - np.min(band_rms)) / (
            np.max(band_rms) - np.min(band_rms) + 1e-6
        )

        adjusted_band = np.copy(band_audio)

        for i in range(min(len(vocal_intensity), len(band_rms_norm))):
            frame_start = i * self.hop_length
            frame_end = min((i + 1) * self.hop_length, len(adjusted_band))

            if frame_end <= frame_start:
                continue

            frame = adjusted_band[frame_start:frame_end]
            if np.max(np.abs(frame)) > 0:
                scale = vocal_intensity[i] / (band_rms_norm[i] + 1e-6)
                adjusted_band[frame_start:frame_end] *= scale

        # Prevent clipping
        max_val = np.max(np.abs(adjusted_band))
        if max_val > 0.95:
            adjusted_band *= 0.95 / max_val

        print("  ✓ Band dynamics adjusted to match vocal")

        return adjusted_band
