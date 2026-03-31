"""
Voice Message Handler for Telegram ACP Bot

This module contains ALL the logic for handling voice messages.
It's designed to be called from the patched bot.py with a SINGLE function call.

Usage in bot.py (patch):
    from audio_transcriber.voice_handler import handle_voice
    text = await handle_voice(message, context, text)

This way, the bot.py patch is MINIMAL and survives updates better.
"""

import logging
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def handle_voice(
    message: "Message",
    context: ContextTypes.DEFAULT_TYPE,
    text: str
) -> str:
    """
    Handle voice message received from Telegram.
    
    This function:
    1. Checks if the message contains a voice note
    2. Downloads the audio file
    3. Transcribes it using the audio_transcriber module
    4. Combines with existing text
    
    Args:
        message: Telegram message object
        context: Telegram bot context
        text: Original message text (may be empty)
    
    Returns:
        Combined text (original + transcription if voice message)
    """
    # Check if message contains a voice note
    if not (message.voice and message.voice.file_id):
        return text
    
    try:
        logger.info("Voice message detected, transcribing...")
        
        # Download the voice message
        file = await context.bot.get_file(message.voice.file_id)
        audio_data = await file.download_as_bytearray()
        
        # Transcribe using the main transcriber module
        from audio_transcriber import transcribe_audio, TranscriptionError
        
        audio_text = await transcribe_audio(bytes(audio_data))
        
        # Handle result
        if audio_text:
            logger.info(f"Transcription successful ({len(audio_text)} chars)")
            result = f"[Trascrizione audio]: {audio_text}"
        else:
            logger.warning("Transcription returned empty text")
            result = "[Audio ricevuto ma nessun contenuto rilevato]"
        
        # Combine with existing text
        if text and result:
            return f"{text}\n\n{result}"
        return result
    
    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e}")
        error_msg = f"[Errore trascrizione audio: {e}]"
        if text:
            return f"{text}\n\n{error_msg}"
        return error_msg
    
    except Exception as e:
        logger.error(f"Voice message handling error: {e}")
        error_msg = "[Errore nella gestione del messaggio vocale]"
        if text:
            return f"{text}\n\n{error_msg}"
        return error_msg
