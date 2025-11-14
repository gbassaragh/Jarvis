"""
JARVIS - Just A Rather Very Intelligent System
Voice interface for AI Assistant Pro

Provides speech-to-text and text-to-speech capabilities for natural voice interaction.
"""

import torch
import numpy as np
from typing import Optional, Generator, Callable
from pathlib import Path
import wave
import threading
import queue

from ai_assistant_pro.utils.logging import get_logger

logger = get_logger("jarvis.voice")


class VoiceInterface:
    """
    Voice interface for JARVIS

    Provides:
    - Speech-to-text (Whisper)
    - Text-to-speech (Bark/XTTS)
    - Wake word detection
    - Continuous listening
    """

    def __init__(
        self,
        stt_model: str = "openai/whisper-base",
        tts_model: str = "suno/bark",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wake_word: str = "jarvis",
    ):
        """
        Initialize voice interface

        Args:
            stt_model: Speech-to-text model
            tts_model: Text-to-speech model
            device: Device for inference
            wake_word: Wake word for activation
        """
        self.device = device
        self.wake_word = wake_word.lower()

        logger.info(f"Initializing JARVIS voice interface on {device}")

        # Initialize STT (Whisper)
        self._init_stt(stt_model)

        # Initialize TTS
        self._init_tts(tts_model)

        # Audio queue for streaming
        self.audio_queue = queue.Queue()
        self.listening = False

        logger.info("✓ JARVIS voice interface ready")

    def _init_stt(self, model_name: str):
        """Initialize speech-to-text model"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            logger.info(f"Loading Whisper model: {model_name}")
            self.stt_processor = WhisperProcessor.from_pretrained(model_name)
            self.stt_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.stt_model = self.stt_model.to(self.device)
            self.stt_model.eval()

            logger.info("✓ Speech-to-text model loaded")
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            self.stt_model = None

    def _init_tts(self, model_name: str):
        """Initialize text-to-speech model"""
        try:
            if "bark" in model_name.lower():
                from transformers import AutoProcessor, BarkModel

                logger.info(f"Loading Bark TTS model: {model_name}")
                self.tts_processor = AutoProcessor.from_pretrained(model_name)
                self.tts_model = BarkModel.from_pretrained(model_name)
                self.tts_model = self.tts_model.to(self.device)
                self.tts_model.eval()
            else:
                logger.warning("TTS model not recognized, using default")
                self.tts_model = None

            logger.info("✓ Text-to-speech model loaded")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self.tts_model = None

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text

        Args:
            audio_path: Path to audio file
            audio_array: Audio array (16kHz)
            sample_rate: Sample rate of audio

        Returns:
            Transcribed text
        """
        if self.stt_model is None:
            return "[STT not available]"

        # Load audio
        if audio_path:
            audio_array, sample_rate = self._load_audio(audio_path)
        elif audio_array is None:
            raise ValueError("Either audio_path or audio_array must be provided")

        # Process audio
        inputs = self.stt_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.stt_model.generate(inputs.input_features)

        # Decode
        transcription = self.stt_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        logger.info(f"Transcribed: {transcription}")
        return transcription

    def speak(
        self,
        text: str,
        voice_preset: str = "v2/en_speaker_6",
        output_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert text to speech

        Args:
            text: Text to speak
            voice_preset: Voice preset to use
            output_path: Optional path to save audio

        Returns:
            Audio array if output_path is None
        """
        if self.tts_model is None:
            logger.warning("TTS not available")
            return None

        logger.info(f"Speaking: {text[:50]}...")

        # Process text
        inputs = self.tts_processor(
            text,
            voice_preset=voice_preset,
            return_tensors="pt"
        ).to(self.device)

        # Generate speech
        with torch.no_grad():
            audio_array = self.tts_model.generate(**inputs)

        audio_array = audio_array.cpu().numpy().squeeze()

        # Save if requested
        if output_path:
            self._save_audio(audio_array, output_path)
            logger.info(f"✓ Saved audio to {output_path}")
            return None

        return audio_array

    def listen_continuous(
        self,
        callback: Callable[[str], str],
        duration: int = 5,
    ):
        """
        Listen continuously for voice commands

        Args:
            callback: Function to process transcribed text
            duration: Duration to listen for each command (seconds)
        """
        logger.info(f"Starting continuous listening (wake word: '{self.wake_word}')")

        try:
            import pyaudio

            # Audio configuration
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000

            p = pyaudio.PyAudio()

            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            logger.info("🎤 Listening...")
            self.listening = True

            while self.listening:
                # Record audio
                frames = []
                for _ in range(0, int(RATE / CHUNK * duration)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

                # Transcribe
                text = self.transcribe(audio_array=audio_data, sample_rate=RATE)

                # Check for wake word
                if self.wake_word in text.lower():
                    logger.info(f"✓ Wake word detected: {text}")

                    # Remove wake word
                    command = text.lower().replace(self.wake_word, "").strip()

                    if command:
                        # Process command
                        response = callback(command)

                        # Speak response
                        if response:
                            self.speak(response)

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
            self.listening = False

    def stop_listening(self):
        """Stop continuous listening"""
        logger.info("Stopping listening")
        self.listening = False

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio file"""
        import librosa
        audio, sr = librosa.load(path, sr=16000)
        return audio, sr

    def _save_audio(self, audio: np.ndarray, path: str, sample_rate: int = 24000):
        """Save audio to WAV file"""
        import scipy.io.wavfile as wavfile

        # Convert to int16
        audio = (audio * 32767).astype(np.int16)

        # Save
        wavfile.write(path, sample_rate, audio)


class WakeWordDetector:
    """
    Lightweight wake word detector

    Uses a small model for efficient always-on detection.
    """

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower()
        logger.info(f"Wake word detector initialized: '{wake_word}'")

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect wake word in audio

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            True if wake word detected
        """
        # Simplified implementation - in production, use a dedicated model
        # like Porcupine or Snowboy
        return False  # Placeholder


class VoiceAssistant:
    """
    Complete voice assistant combining voice interface with AI engine
    """

    def __init__(
        self,
        engine,
        voice_interface: Optional[VoiceInterface] = None,
        wake_word: str = "jarvis",
    ):
        """
        Initialize voice assistant

        Args:
            engine: AssistantEngine instance
            voice_interface: VoiceInterface instance (creates one if None)
            wake_word: Wake word for activation
        """
        self.engine = engine

        if voice_interface is None:
            self.voice = VoiceInterface(wake_word=wake_word)
        else:
            self.voice = voice_interface

        logger.info("✓ JARVIS voice assistant ready")

    def process_voice_command(self, command: str) -> str:
        """
        Process voice command

        Args:
            command: Voice command text

        Returns:
            Response text
        """
        logger.info(f"Processing command: {command}")

        # Generate response using engine
        response = self.engine.generate(
            prompt=command,
            max_tokens=150,
            temperature=0.7,
        )

        return response

    def start(self):
        """Start voice assistant"""
        logger.info("🚀 Starting JARVIS...")

        def callback(command: str) -> str:
            return self.process_voice_command(command)

        self.voice.listen_continuous(callback=callback)

    def stop(self):
        """Stop voice assistant"""
        self.voice.stop_listening()
        logger.info("JARVIS stopped")


# Example usage
if __name__ == "__main__":
    from ai_assistant_pro import AssistantEngine

    # Create engine
    engine = AssistantEngine(model_name="gpt2")

    # Create voice assistant
    jarvis = VoiceAssistant(engine=engine, wake_word="jarvis")

    # Start listening
    try:
        jarvis.start()
    except KeyboardInterrupt:
        jarvis.stop()
