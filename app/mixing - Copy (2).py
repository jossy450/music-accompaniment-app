# In app/mixing.py - ROBUST VERSION (mono-safe, vectorised)

from __future__ import annotations
import numpy as np
import librosa


class ProfessionalMixMaster:
    """
    Mixing + mastering engine used by the main app.

    Public API:
        - mix_tracks(vocal, accompaniment, vocal_level_db, music_level_db, mixing_profile)
        - master(audio, apply_stereo_widening=True)

    All processing is mono internally. Stereo widening is applied only at the end,
    so the rest of the pipeline never sees (N, 2) arrays and we avoid ambiguous
    boolean errors.
    """

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.target_loudness = -14.0  # pseudo-LUFS target

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def mix_tracks(
        self,
        vocal: np.ndarray,
        accompaniment: np.ndarray,
        vocal_level_db: float = 0.0,
        music_level_db: float = -6.0,
        mixing_profile: str = "vocal_forward",
    ) -> np.ndarray:
        """
        Mix vocal and accompaniment.

        mixing_profile:
            - 'vocal_forward'  : vocal slightly pushed, band slightly down
            - 'balanced'       : roughly equal
        """
        vocal = self._ensure_mono(vocal)
        accompaniment = self._ensure_mono(accompaniment)

        max_len = max(len(vocal), len(accompaniment))
        vocal = self._pad_or_trim(vocal, max_len)
        accompaniment = self._pad_or_trim(accompaniment, max_len)

        # Simple EQ tilts to keep the “professional” flavour
        vocal = self._presence_eq(vocal)
        accompaniment = self._warmth_eq(accompaniment)

        # Per-track compression
        vocal = self._compress(vocal, threshold_db=-18.0, ratio=3.0)
        accompaniment = self._compress(accompaniment, threshold_db=-20.0, ratio=2.5)

        # Level staging
        vocal_gain = self._db_to_lin(vocal_level_db)
        music_gain = self._db_to_lin(music_level_db)

        vocal *= vocal_gain
        accompaniment *= music_gain

        if mixing_profile == "vocal_forward":
            mixed = vocal + accompaniment * 0.7
        elif mixing_profile == "balanced":
            mixed = 0.5 * (vocal + accompaniment)
        else:
            mixed = vocal + accompaniment

        # Pre-master safety normalize
        peak = np.max(np.abs(mixed)) + 1e-12
        if peak > 1.0:
            mixed = mixed / peak

        return mixed.astype(np.float32)

    def master(self, audio: np.ndarray, apply_stereo_widening: bool = True) -> np.ndarray:
        """
        Mastering chain:
            1. Multiband-ish tilt via spectral shaping
            2. Broadband compression
            3. Soft limiter
            4. Loudness normalization to ~-14 LUFS
            5. Optional stereo widening
        """
        audio = self._ensure_mono(audio).astype(np.float32)

        # Gentle EQ
        audio = self._tilt_eq(audio)

        # Broadband compression
        audio = self._compress(audio, threshold_db=-18.0, ratio=3.0)

        # Limiter
        audio = self._limit(audio, threshold_db=-0.5)

        # Loudness normalisation
        audio = self._normalize_loudness(audio, target_lufs=self.target_loudness)

        # Final safety limiter
        audio = self._limit(audio, threshold_db=-0.5)

        # Stereo widen at the end (optional)
        if apply_stereo_widening:
            audio = self._stereo_widen(audio)
        else:
            audio = audio[:, None]  # mono → (N, 1)

        return audio.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_mono(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x
        return np.mean(x, axis=1)

    def _pad_or_trim(self, x: np.ndarray, length: int) -> np.ndarray:
        if len(x) == length:
            return x
        if len(x) > length:
            return x[:length]
        pad = length - len(x)
        return np.pad(x, (0, pad), mode="constant")

    def _db_to_lin(self, db: float) -> float:
        return float(10.0 ** (db / 20.0))

    # ----- EQ helpers -------------------------------------------------- #

    def _presence_eq(self, audio: np.ndarray) -> np.ndarray:
        """Very light presence boost (around 2–4 kHz)."""
        n_fft = 2048
        hop = 512
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # +3 dB around 2–4kHz, fading outside
        gain_db = np.zeros_like(freqs)
        mask = (freqs >= 2000) & (freqs <= 4000)
        gain_db[mask] = 3.0
        gain_lin = 10.0 ** (gain_db / 20.0)

        S_eq = S * gain_lin[:, None]
        out = librosa.istft(S_eq, hop_length=hop, length=len(audio))
        return out.astype(np.float32)

    def _warmth_eq(self, audio: np.ndarray) -> np.ndarray:
        """Subtle warmth boost (~100–300 Hz)."""
        n_fft = 2048
        hop = 512
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        gain_db = np.zeros_like(freqs)
        mask = (freqs >= 80) & (freqs <= 300)
        gain_db[mask] = 2.0
        gain_lin = 10.0 ** (gain_db / 20.0)

        S_eq = S * gain_lin[:, None]
        out = librosa.istft(S_eq, hop_length=hop, length=len(audio))
        return out.astype(np.float32)

    def _tilt_eq(self, audio: np.ndarray) -> np.ndarray:
        """Gentle low cut / high lift tilt."""
        n_fft = 2048
        hop = 512
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # -2 dB at 50 Hz → +2 dB at 10 kHz
        low_ref, high_ref = 50.0, 10000.0
        low_db, high_db = -2.0, 2.0

        f = np.clip(freqs, 1.0, None)
        pos = (np.log10(f) - np.log10(low_ref)) / (np.log10(high_ref) - np.log10(low_ref))
        pos = np.clip(pos, 0.0, 1.0)
        tilt_db = low_db + (high_db - low_db) * pos
        tilt_lin = 10.0 ** (tilt_db / 20.0)

        S_tilt = S * tilt_lin[:, None]
        out = librosa.istft(S_tilt, hop_length=hop, length=len(audio))
        return out.astype(np.float32)

    # ----- Compression / limiting (vectorised) ------------------------- #

    def _compress(
        self,
        audio: np.ndarray,
        threshold_db: float = -18.0,
        ratio: float = 3.0,
    ) -> np.ndarray:
        """Vectorised broadband compressor."""
        eps = 1e-12
        mag = np.abs(audio) + eps
        audio_db = 20.0 * np.log10(mag)

        over = audio_db > threshold_db
        gain_db = np.zeros_like(audio_db)
        gain_db[over] = threshold_db + (audio_db[over] - threshold_db) / ratio - audio_db[over]

        gain_lin = 10.0 ** (gain_db / 20.0)
        return (audio * gain_lin).astype(np.float32)

    def _limit(self, audio: np.ndarray, threshold_db: float = -0.5) -> np.ndarray:
        """Soft limiter around a given peak threshold."""
        eps = 1e-12
        mag = np.abs(audio) + eps
        audio_db = 20.0 * np.log10(mag)

        over = audio_db > threshold_db
        gain_db = np.zeros_like(audio_db)
        gain_db[over] = (threshold_db - audio_db[over]) * 0.5  # soft knee

        gain_lin = 10.0 ** (gain_db / 20.0)
        limited = audio * gain_lin

        thresh_lin = self._db_to_lin(threshold_db)
        limited = np.clip(limited, -thresh_lin, thresh_lin)
        return limited.astype(np.float32)

    # ----- Loudness / stereo ------------------------------------------ #

    def _estimate_lufs(self, audio: np.ndarray) -> float:
        """Rough RMS-based LUFS estimate."""
        eps = 1e-12
        rms = np.sqrt(np.mean(audio**2) + eps)
        return float(20.0 * np.log10(rms + eps))

    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        current = self._estimate_lufs(audio)
        diff_db = target_lufs - current
        gain = self._db_to_lin(diff_db)
        out = audio * gain
        peak = np.max(np.abs(out)) + 1e-12
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)

    def _stereo_widen(self, mono: np.ndarray, delay_ms: float = 12.0) -> np.ndarray:
        """Simple Haas-based stereo widening."""
        delay_samples = int(self.sr * delay_ms / 1000.0)
        if delay_samples <= 0:
            return mono[:, None].astype(np.float32)

        left = mono
        right = np.pad(mono, (delay_samples, 0), mode="constant")[: len(mono)]

        stereo = np.stack([left, right], axis=1)
        peak = np.max(np.abs(stereo)) + 1e-12
        if peak > 1.0:
            stereo = stereo / peak
        return stereo.astype(np.float32)
