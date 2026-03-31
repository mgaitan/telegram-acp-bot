"""
Base Transcriber Module

Abstract base class for all transcription providers.
All providers must inherit from this class and implement the transcribe method.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        self.message = message
        self.provider = provider
        super().__init__(self.message)


class BaseTranscriber(ABC):
    """
    Abstract base class for audio transcription providers.
    
    This class defines the interface that all transcription providers must implement.
    To add a new provider, inherit from this class and implement the transcribe method.
    
    Example:
        class MyProviderTranscriber(BaseTranscriber):
            async def transcribe(self, audio_data: bytes, language: str = "it") -> str:
                # Implementation here
                pass
    """
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the transcriber.
        
        Args:
            api_key: API key for the transcription service
            model: Model name to use for transcription
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self._initialized = False
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = "it") -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio file bytes (max 25MB)
            language: Language code for transcription (default: "it" for Italian)
        
        Returns:
            Transcribed text as string
        
        Raises:
            TranscriptionError: If transcription fails
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize the transcriber (optional, can be overridden by providers).
        
        Returns:
            True if initialization successful, False otherwise
        """
        self._initialized = True
        return True
    
    async def close(self):
        """
        Close any open connections (optional, can be overridden by providers).
        """
        self._initialized = False
    
    def is_available(self) -> bool:
        """
        Check if the transcriber is available and ready to use.
        
        Returns:
            True if available, False otherwise
        """
        return self._initialized
    
    @property
    def provider_name(self) -> str:
        """
        Return the name of the provider.
        
        Returns:
            Provider name (e.g., "groq", "openai", "azure")
        """
        return self.__class__.__name__.replace("Transcriber", "").lower()
    
    def __str__(self) -> str:
        return f"{self.provider_name} transcriber ({self.model})"
