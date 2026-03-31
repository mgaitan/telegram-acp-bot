"""
Audio Transcriber Main Module

Main entry point for audio transcription functionality.
Handles provider selection, initialization, and transcription requests.
"""

import asyncio
import logging
from typing import Optional, Type
from .config import TranscriberConfig, get_config
from .providers.base import BaseTranscriber, TranscriptionError
from .providers.groq import GroqTranscriber

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Main audio transcriber class.
    
    Manages transcription providers and handles transcription requests.
    Supports multiple providers with automatic fallback.
    
    Example:
        transcriber = AudioTranscriber()
        await transcriber.initialize()
        
        text = await transcriber.transcribe(audio_bytes)
        
        await transcriber.close()
    """
    
    # Provider registry - add new providers here
    PROVIDERS: dict[str, Type[BaseTranscriber]] = {
        "groq": GroqTranscriber,
        # Add more providers here:
        # "openai": OpenAITranscriber,
        # "azure": AzureTranscriber,
    }
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        """
        Initialize the audio transcriber.
        
        Args:
            config: Optional configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self._transcriber: Optional[BaseTranscriber] = None
        self._fallback_transcriber: Optional[BaseTranscriber] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the transcription service.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.debug("Transcriber already initialized")
            return True
        
        if not self.config.enabled:
            logger.info("Audio transcription is disabled")
            return False
        
        try:
            # Initialize primary transcriber
            self._transcriber = await self._create_transcriber(
                self.config.provider,
                self._get_api_key(self.config.provider)
            )
            
            if self._transcriber:
                await self._transcriber.initialize()
                logger.info(f"Primary transcriber initialized: {self.config.provider}")
            
            # Initialize fallback if enabled
            if self.config.fallback_enabled and self.config.fallback_provider:
                if self.config.fallback_provider != self.config.provider:
                    self._fallback_transcriber = await self._create_transcriber(
                        self.config.fallback_provider,
                        self._get_api_key(self.config.fallback_provider)
                    )
                    if self._fallback_transcriber:
                        await self._fallback_transcriber.initialize()
                        logger.info(
                            f"Fallback transcriber initialized: "
                            f"{self.config.fallback_provider}"
                        )
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            return False
    
    async def _create_transcriber(
        self, 
        provider: str, 
        api_key: Optional[str]
    ) -> Optional[BaseTranscriber]:
        """
        Create a transcriber instance for the specified provider.
        
        Args:
            provider: Provider name ("groq", "openai", etc.)
            api_key: API key for the provider
        
        Returns:
            Transcriber instance or None if not available
        """
        if provider not in self.PROVIDERS:
            logger.error(f"Unknown provider: {provider}")
            return None
        
        if not api_key:
            logger.error(f"No API key for provider: {provider}")
            return None
        
        transcriber_class = self.PROVIDERS[provider]
        return transcriber_class(
            api_key=api_key,
            model=self.config.whisper_model,
            timeout=self.config.timeout,
            retries=self.config.retries,
        )
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider."""
        if provider == "groq":
            return self.config.groq_api_key
        elif provider == "openai":
            return self.config.openai_api_key
        return None
    
    async def transcribe(
        self, 
        audio_data: bytes, 
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio file bytes
            language: Optional language code (uses config default if None)
        
        Returns:
            Transcribed text
        
        Raises:
            TranscriptionError: If transcription fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.config.enabled:
            raise TranscriptionError(
                "Audio transcription is disabled",
                provider="config"
            )
        
        if not self._transcriber:
            raise TranscriptionError(
                "No transcriber available",
                provider=self.config.provider
            )
        
        # Use provided language or default
        lang = language or self.config.language
        
        # Try primary transcriber
        try:
            logger.debug(f"Transcribing audio with {self.config.provider}")
            return await self._transcriber.transcribe(audio_data, lang)
        
        except TranscriptionError as e:
            logger.warning(f"Primary transcription failed: {e}")
            
            # Try fallback if available
            if self._fallback_transcriber:
                try:
                    logger.info("Trying fallback transcriber")
                    return await self._fallback_transcriber.transcribe(audio_data, lang)
                
                except TranscriptionError as fallback_error:
                    logger.error(f"Fallback transcription also failed: {fallback_error}")
            
            # Re-raise original error
            raise
    
    async def close(self):
        """Close all transcribers and cleanup resources."""
        if self._transcriber:
            await self._transcriber.close()
            self._transcriber = None
        
        if self._fallback_transcriber:
            await self._fallback_transcriber.close()
            self._fallback_transcriber = None
        
        self._initialized = False
        logger.info("Audio transcriber closed")
    
    @property
    def is_available(self) -> bool:
        """Check if transcription service is available."""
        return self._initialized and self._transcriber is not None
    
    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.config.provider
    
    def get_status(self) -> dict:
        """
        Get current status of the transcription service.
        
        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "provider": self.config.provider,
            "model": self.config.whisper_model,
            "language": self.config.language,
            "available": self.is_available,
            "fallback_enabled": self.config.fallback_enabled,
            "fallback_provider": self.config.fallback_provider,
        }


# Global transcriber instance (lazy loaded)
_transcriber: Optional[AudioTranscriber] = None


def get_transcriber() -> AudioTranscriber:
    """
    Get the global transcriber instance.
    
    Returns:
        AudioTranscriber instance
    """
    global _transcriber
    if _transcriber is None:
        _transcriber = AudioTranscriber()
    return _transcriber


async def initialize_transcriber() -> bool:
    """
    Initialize the global transcriber.
    
    Returns:
        True if initialization successful
    """
    transcriber = get_transcriber()
    return await transcriber.initialize()


async def transcribe_audio(
    audio_data: bytes, 
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio using the global transcriber.
    
    Args:
        audio_data: Raw audio file bytes
        language: Optional language code
    
    Returns:
        Transcribed text
    """
    transcriber = get_transcriber()
    return await transcriber.transcribe(audio_data, language)


async def close_transcriber():
    """Close the global transcriber."""
    global _transcriber
    if _transcriber:
        await _transcriber.close()
        _transcriber = None
