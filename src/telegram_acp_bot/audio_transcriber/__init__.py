"""
Audio Transcriber Module for Telegram ACP Bot

This module provides audio transcription capabilities for the Telegram ACP Bot.
It supports multiple transcription providers with a unified interface.

Quick Start:
    from audio_transcriber import (
        initialize_transcriber,
        transcribe_audio,
        close_transcriber
    )
    
    # Initialize
    await initialize_transcriber()
    
    # Transcribe
    text = await transcribe_audio(audio_bytes)
    
    # Cleanup
    await close_transcriber()

Configuration:
    Set environment variables:
    - AUDIO_TRANSCRIPTION_ENABLED=true
    - AUDIO_TRANSCRIPTION_PROVIDER=groq
    - GROQ_API_KEY=your-api-key
    - WHISPER_MODEL=whisper-large-v3
    - TRANSCRIPTION_LANGUAGE=it

Providers:
    - groq: Groq Whisper API (whisper-large-v3, whisper-large-v3-turbo)
    - openai: OpenAI Whisper API (future)
"""

from .config import TranscriberConfig, get_config, reload_config
from .transcriber import (
    AudioTranscriber,
    get_transcriber,
    initialize_transcriber,
    transcribe_audio,
    close_transcriber,
)
from .providers.base import BaseTranscriber, TranscriptionError
from .providers.groq import GroqTranscriber

__version__ = "1.0.0"
__author__ = "Telegram ACP Bot Audio Extension"

__all__ = [
    # Main functions
    "initialize_transcriber",
    "transcribe_audio",
    "close_transcriber",
    "get_transcriber",
    
    # Configuration
    "get_config",
    "reload_config",
    "TranscriberConfig",
    
    # Provider classes
    "BaseTranscriber",
    "GroqTranscriber",
    
    # Exceptions
    "TranscriptionError",
    
    # Main class
    "AudioTranscriber",
]
