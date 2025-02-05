import logging
import time
from pathlib import Path
from typing import Optional

import torch
from pydub import AudioSegment, playback

from .utils.tools import prepare_text

_LOGGER = logging.getLogger(__name__)


class TTSRunner:
    """Text-to-Speech runner for GLaDOS TTS."""

    def __init__(
        self,
        use_p1: bool = False,
        log: bool = False,
        models_dir: Path = Path('models'),
    ):
        """
        Initialize the TTS engine.

        Args:
            use_p1: Whether to use the 'glados_p1.pt' embedding.
            log: Enable detailed logging.
            models_dir: Directory where model files are stored.
        """
        self.log = log
        self.models_dir = models_dir

        emb_filename = 'glados_p1.pt' if use_p1 else 'glados_p2.pt'
        emb_path = self.models_dir / 'emb' / emb_filename
        if not emb_path.is_file():
            raise FileNotFoundError(f"Embedding model not found at {emb_path}")

        # Select the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # For Apple Silicon
        else:
            self.device = torch.device('cpu')

        _LOGGER.info(f"Using device: {self.device}")

        # Load the embedding
        self.emb = torch.load(str(emb_path), map_location=self.device, weights_only=True)
        self.emb = self.emb.to(self.device)

        # Load models
        glados_model_path = self.models_dir / 'glados-new.pt'
        vocoder_model_path = self.models_dir / 'vocoder-gpu.pt'

        self.glados = torch.jit.load(str(glados_model_path), map_location=self.device)
        self.vocoder = torch.jit.load(str(vocoder_model_path), map_location=self.device)

        self.glados.to(self.device)
        self.vocoder.to(self.device)

        # Warm-up the models
        self._warmup_models()

    def _warmup_models(self):
        """Warm-up the models to reduce initial inference time."""
        _LOGGER.info("Warming up models...")
        with torch.no_grad():
            x = prepare_text("Hello", self.models_dir, self.device)
            x = x.to(self.device)
            self.emb = self.emb.to(self.device)
            outputs = self.glados.generate_jit(x, self.emb, 1.0)
            mel = outputs['mel_post'].to(self.device)
            _ = self.vocoder(mel)
        _LOGGER.info("Models warmed up.")

    def run_tts(self, text: str, alpha: float = 1.0) -> AudioSegment:
        """
        Generate speech audio from text.

        Args:
            text: The input text to synthesize.
            alpha: Speed factor for the TTS.

        Returns:
            An AudioSegment containing the synthesized speech.
        """
        x = prepare_text(text, self.models_dir, self.device)
        x = x.to(self.device)
        self.emb = self.emb.to(self.device)

        with torch.no_grad():
            # Generate TTS output
            if self.log:
                start_time = time.time()
            tts_output = self.glados.generate_jit(x, self.emb, alpha)
            if self.log:
                _LOGGER.debug(f"Forward Tacotron took {(time.time() - start_time) * 1000:.2f} ms")

            # Generate audio waveform
            if self.log:
                start_time = time.time()
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            if self.log:
                _LOGGER.debug(f"HiFiGAN took {(time.time() - start_time) * 1000:.2f} ms")

            # Normalize and convert to AudioSegment
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')

            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=22050,
                sample_width=2,  # 16-bit audio
                channels=1,
            )
            return audio_segment

    def play_audio(self, audio: AudioSegment):
        """
        Play an AudioSegment.

        Args:
            audio: The AudioSegment to play.
        """
        playback.play(audio)

    def speak(self, text: str, alpha: float = 1.0):
        """
        Synthesize and play speech from text.

        Args:
            text: The input text to synthesize.
            alpha: Speed factor for the TTS.
        """
        audio = self.run_tts(text, alpha)
        self.play_audio(audio)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="GLaDOS TTS Runner")
    parser.add_argument('--use_p1', action='store_true', help='Use glados_p1.pt embedding')
    parser.add_argument('--log', action='store_true', help='Enable detailed logging')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory of model files')
    args = parser.parse_args()

    glados = TTSRunner(use_p1=args.use_p1, log=args.log, models_dir=args.models_dir)

    while True:
        try:
            text = input("Input: ")
            if text.strip():
                glados.speak(text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
