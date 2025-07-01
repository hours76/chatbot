import subprocess
import numpy as np
import sounddevice as sd
from config import (
    PIPER_BIN, PIPER_MODEL_EN, PIPER_MODEL_ZH, 
    PIPER_SAMPLE_RATE, PIPER_ESPEAK_DATA, DEBUG_PIPER
)
from utils import pretty_print

class TextToSpeech:
    def __init__(self):
        self.model_map = {
            'en': PIPER_MODEL_EN,
            'zh': PIPER_MODEL_ZH,
        }
    
    def speak(self, text: str, lang_code: str = 'en'):
        """Generate and play speech from text using Piper TTS."""
        lang = lang_code if lang_code in ('en', 'zh') else 'en'
        pretty_print("[PIPER]", f"Generating TTS response with Piper (lang: {lang})...")
        
        try:
            model_path = self._get_model_for_lang(lang)
            cmd = [
                PIPER_BIN,
                "--model", model_path,
                "--espeak_data", PIPER_ESPEAK_DATA,
                "--output_raw"
            ]
            
            if DEBUG_PIPER:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                raw_audio, stderr = proc.communicate(input=text.encode("utf-8"))
                # Print each line of Piper's output with [DEBUG_PIPER] prefix
                for line in stderr.decode(errors='ignore').splitlines():
                    if line.strip():
                        print(f"[DEBUG_PIPER] {line}")
            else:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                raw_audio, _ = proc.communicate(input=text.encode("utf-8"))
            
            audio = np.frombuffer(raw_audio, dtype=np.int16)
            sd.play(audio, PIPER_SAMPLE_RATE)
            sd.wait()
            
        except Exception as e:
            pretty_print("[PIPER]", f"Error during TTS playback: {e}")
    
    def _get_model_for_lang(self, lang_code: str) -> str:
        """Return the Piper model path for a given language code."""
        return self.model_map.get(lang_code, self.model_map['en'])