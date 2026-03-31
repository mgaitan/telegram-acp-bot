"""
Groq Whisper Transcriber Module

Implementation of the BaseTranscriber for Groq's Whisper API.
Supports whisper-large-v3 and whisper-large-v3-turbo models.
"""

import asyncio
import httpx
import logging
from typing import Optional
from .base import BaseTranscriber, TranscriptionError

logger = logging.getLogger(__name__)

# Groq API endpoint for Whisper
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Supported models
GROQ_WHISPER_MODELS = {
    "whisper-large-v3": {
        "description": "OpenAI Whisper Large v3 - Maximum accuracy",
        "max_file_size_mb": 25,
        "supported_languages": "99+ languages including Italian",
    },
    "whisper-large-v3-turbo": {
        "description": "OpenAI Whisper Large v3 Turbo - Faster, slightly less accurate",
        "max_file_size_mb": 25,
        "supported_languages": "99+ languages including Italian",
    },
}

# Rate limits (from Groq documentation)
GROQ_RATE_LIMITS = {
    "requests_per_minute": 30,
    "requests_per_day": 14400,
    "max_file_size_mb": 25,
}


class GroqTranscriber(BaseTranscriber):
    """
    Groq Whisper transcription provider.
    
    Uses Groq's API to transcribe audio using OpenAI's Whisper models.
    Supports both whisper-large-v3 and whisper-large-v3-turbo.
    
    Rate Limits:
        - 30 requests per minute
        - 14,400 requests per day
        - 25 MB max file size
    
    Example:
        transcriber = GroqTranscriber(
            api_key="your-api-key",
            model="whisper-large-v3"
        )
        text = await transcriber.transcribe(audio_bytes, language="it")
    """
    
    def __init__(self, api_key: str, model: str = "whisper-large-v3", **kwargs):
        """
        Initialize Groq transcriber.
        
        Args:
            api_key: Groq API key
            model: Whisper model to use ("whisper-large-v3" or "whisper-large-v3-turbo")
            **kwargs: Additional configuration (timeout, retries, etc.)
        """
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        self.timeout = kwargs.get("timeout", 60)  # Default 60 seconds
        self.retries = kwargs.get("retries", 3)  # Default 3 retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> bool:
        """
        Initialize the HTTP client.
        
        Returns:
            True if initialization successful
        """
        try:
            if self._client is None:
                timeout = httpx.Timeout(self.timeout)
                self._client = httpx.AsyncClient(timeout=timeout)
            
            # Validate model
            if self.model not in GROQ_WHISPER_MODELS:
                logger.warning(
                    f"Model '{self.model}' not in known models. "
                    f"Available: {list(GROQ_WHISPER_MODELS.keys())}"
                )
            
            self._initialized = True
            logger.info(f"Groq transcriber initialized with model: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq transcriber: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("Groq transcriber closed")
    
    async def transcribe(self, audio_data: bytes, language: str = "it") -> str:
        """
        Transcribe audio data using Groq Whisper API.
        
        Args:
            audio_data: Raw audio file bytes (max 25MB)
            language: Language code (default: "it" for Italian)
        
        Returns:
            Transcribed text as string
        
        Raises:
            TranscriptionError: If transcription fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate audio size
        max_size = GROQ_RATE_LIMITS["max_file_size_mb"] * 1024 * 1024
        if len(audio_data) > max_size:
            raise TranscriptionError(
                f"Audio file too large: {len(audio_data) / (1024*1024):.2f}MB "
                f"(max: {GROQ_RATE_LIMITS['max_file_size_mb']}MB)",
                provider="groq"
            )
        
        # Prepare the request
        files = {
            "file": ("audio.wav", audio_data, "audio/wav"),
            "model": (None, self.model),
            "language": (None, language),
            "response_format": (None, "json"),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Execute with retries
        last_error = None
        for attempt in range(self.retries):
            try:
                logger.debug(f"Transcription attempt {attempt + 1}/{self.retries}")
                
                response = await self._client.post(
                    GROQ_WHISPER_URL,
                    headers=headers,
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    
                    if not text:
                        logger.warning("Transcription returned empty text")
                        return "[Audio ricevuto ma nessun contenuto rilevato]"
                    
                    logger.info(f"Transcription successful ({len(text)} chars)")
                    return text
                
                elif response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                else:
                    error_text = response.text[:200]
                    logger.error(f"Groq API error: {response.status_code} - {error_text}")
                    
                    if response.status_code >= 500:
                        # Server error, retry
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # Client error, don't retry
                        raise TranscriptionError(
                            f"Groq API error: {response.status_code} - {error_text}",
                            provider="groq"
                        )
            
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # All retries failed
        error_msg = f"Transcription failed after {self.retries} attempts"
        if last_error:
            error_msg += f": {last_error}"
        
        logger.error(error_msg)
        raise TranscriptionError(error_msg, provider="groq")
    
    @property
    def rate_limits(self) -> dict:
        """Return current Groq rate limits."""
        return GROQ_RATE_LIMITS.copy()
    
    @property
    def model_info(self) -> Optional[dict]:
        """Return information about the current model."""
        return GROQ_WHISPER_MODELS.get(self.model)
    
    def __str__(self) -> str:
        model_info = self.model_info
        if model_info:
            return f"Groq transcriber ({self.model}) - {model_info['description']}"
        return f"Groq transcriber ({self.model})"
