"""
Audio Transcriber Configuration Module

Handles loading and validating configuration from environment variables.
Supports multiple providers and fallback configurations.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TranscriberConfig:
    """Configuration for the audio transcriber."""
    
    # Enable/disable transcription
    enabled: bool = True
    
    # Provider selection ("groq", "openai", etc.)
    provider: str = "groq"
    
    # API keys for different providers
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Model configuration
    whisper_model: str = "whisper-large-v3"
    
    # Transcription settings
    language: str = "it"  # Default to Italian
    timeout: int = 60  # Seconds
    retries: int = 3
    
    # File limits
    max_file_size_mb: int = 25
    
    # Debug mode
    debug: bool = False
    
    # Fallback settings
    fallback_enabled: bool = False
    fallback_provider: Optional[str] = None
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "TranscriberConfig":
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file (not used, relies on os.environ)
        
        Returns:
            TranscriberConfig instance
        """
        # Helper to get boolean from env
        def get_bool(name: str, default: bool) -> bool:
            value = os.getenv(name, str(default)).lower()
            return value in ("true", "1", "yes", "on")
        
        # Helper to get optional string
        def get_opt_str(name: str) -> Optional[str]:
            value = os.getenv(name, "").strip()
            return value if value else None
        
        config = cls(
            enabled=get_bool("AUDIO_TRANSCRIPTION_ENABLED", True),
            provider=os.getenv("AUDIO_TRANSCRIPTION_PROVIDER", "groq").lower(),
            groq_api_key=get_opt_str("GROQ_API_KEY"),
            openai_api_key=get_opt_str("OPENAI_API_KEY"),
            whisper_model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            language=os.getenv("TRANSCRIPTION_LANGUAGE", "it"),
            timeout=int(os.getenv("TRANSCRIPTION_TIMEOUT", "60")),
            retries=int(os.getenv("TRANSCRIPTION_RETRIES", "3")),
            max_file_size_mb=int(os.getenv("AUDIO_MAX_FILE_SIZE_MB", "25")),
            debug=get_bool("AUDIO_TRANSCRIPTION_DEBUG", False),
            fallback_enabled=get_bool("AUDIO_FALLBACK_ENABLED", False),
            fallback_provider=get_opt_str("AUDIO_FALLBACK_PROVIDER"),
        )
        
        # Validate configuration
        config._validate()
        
        return config
    
    def _validate(self):
        """Validate the configuration."""
        # Check API key for selected provider
        if self.enabled:
            if self.provider == "groq" and not self.groq_api_key:
                logger.warning(
                    "Audio transcription enabled but GROQ_API_KEY not set. "
                    "Transcription will fail."
                )
            elif self.provider == "openai" and not self.openai_api_key:
                logger.warning(
                    "Audio transcription enabled but OPENAI_API_KEY not set. "
                    "Transcription will fail."
                )
        
        # Validate provider
        valid_providers = ["groq", "openai"]
        if self.provider not in valid_providers:
            logger.warning(
                f"Unknown provider '{self.provider}'. Using 'groq' as default."
            )
            self.provider = "groq"
        
        # Validate model
        valid_models = ["whisper-large-v3", "whisper-large-v3-turbo"]
        if self.whisper_model not in valid_models:
            logger.warning(
                f"Unknown model '{self.whisper_model}'. "
                f"Valid models: {valid_models}"
            )
        
        # Validate language code
        if len(self.language) != 2:
            logger.warning(
                f"Language code '{self.language}' may be invalid. "
                "Expected 2-letter ISO code (e.g., 'it', 'en')."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "whisper_model": self.whisper_model,
            "language": self.language,
            "timeout": self.timeout,
            "retries": self.retries,
            "max_file_size_mb": self.max_file_size_mb,
            "debug": self.debug,
            "fallback_enabled": self.fallback_enabled,
            "fallback_provider": self.fallback_provider,
            # Note: API keys are NOT included for security
        }
    
    def __str__(self) -> str:
        return (
            f"TranscriberConfig(provider={self.provider}, "
            f"model={self.whisper_model}, language={self.language})"
        )


# Global configuration instance (lazy loaded)
_config: Optional[TranscriberConfig] = None


def get_config() -> TranscriberConfig:
    """
    Get the global configuration instance.
    
    Returns:
        TranscriberConfig instance
    """
    global _config
    if _config is None:
        _config = TranscriberConfig.from_env()
    return _config


def reload_config() -> TranscriberConfig:
    """
    Reload configuration from environment variables.
    
    Returns:
        New TranscriberConfig instance
    """
    global _config
    _config = TranscriberConfig.from_env()
    logger.info("Configuration reloaded")
    return _config
