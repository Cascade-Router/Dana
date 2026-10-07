"""Voice I/O for the API server: speech-to-text (``stt``, fed by
``devices``' input-device selection, driven by
``dana.services.voice_service``) and spoken replies (``multi_voice_tts``,
Piper via ``tts_worker`` with a pyttsx3 fallback, used by
``dana.api.server._speak_reply``)."""
