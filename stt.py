import re
import subprocess
from langdetect import detect
from config import WHISPER_MODEL
from utils import whisper_print, pretty_print

class SpeechToText:
    def __init__(self, lang_code: str = ""):
        self.lang_code = lang_code
    
    def transcribe(self, filepath: str) -> tuple[str, str]:
        """
        Transcribe audio file using Whisper.
        Returns (transcript, detected_language)
        """
        pretty_print("[WHISPER]", "Running whisper-cpp transcription...")
        try:
            cmd = ["whisper-cpp", "--model", WHISPER_MODEL, "--file", filepath]
            if self.lang_code:
                cmd.extend(["-l", self.lang_code])
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Show raw Whisper output (prefix each line)
            for ln in result.stdout.splitlines():
                whisper_print(ln)
            if result.stderr.strip():
                for ln in result.stderr.splitlines():
                    whisper_print(ln)
            
            lines = result.stdout.strip().splitlines()
            lines = [re.sub(r"\[.*?\]\s*", "", line) for line in lines if line.strip()]
            transcript = " ".join(lines)
            pretty_print("[WHISPER]", transcript)
            
            # Use langdetect to detect language from transcript
            detected_lang = self._detect_language(transcript)
            pretty_print("[LANGDETECT]", f"Whisper transcript language detected: {detected_lang}")
            
            return transcript, detected_lang
            
        except Exception as e:
            whisper_print("Whisper error:", e)
            return "", 'en'
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text, defaulting to English."""
        try:
            detected_lang = detect(text)
            if detected_lang.startswith('zh'):
                return 'zh'
            else:
                return 'en'
        except Exception:
            return 'en'