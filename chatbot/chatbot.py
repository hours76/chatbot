import os
import sys
import subprocess
import argparse
import warnings

# ---- Suppress pkg_resources deprecation warning (Python 3.12+) ----
if sys.version_info >= (3, 12):
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources.*deprecated",
        category=UserWarning,
        module="webrtcvad",
    )

from audio import AudioRecorder
from stt import SpeechToText
from tts import TextToSpeech
from llm import LLMClient
from config import MIN_DURATION, OUTFILE, DEFAULT_PROMPTS
from utils import pretty_print

class ChatBot:
    def __init__(self, lang_code: str = "", custom_prompt: str = None):
        self.recorder = AudioRecorder()
        self.stt = SpeechToText(lang_code)
        self.tts = TextToSpeech()
        self.llm = LLMClient()
        self.custom_prompt = custom_prompt
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def handle_transcript(self, transcript_data):
        """Process transcript and generate response."""
        transcript, detected_lang = transcript_data if isinstance(transcript_data, tuple) else (transcript_data, None)
        
        # Set prompt based on detected language if not specified
        if self.custom_prompt:
            prompt = self.custom_prompt
        else:
            prompt = DEFAULT_PROMPTS.get(detected_lang, DEFAULT_PROMPTS['en'])
        
        full_prompt = f"{transcript} {prompt}".strip()
        pretty_print("[OLLAMA]", f"System prompt: {prompt}")
        
        reply = self.llm.generate_response(full_prompt)
        pretty_print("[OLLAMA]", reply)
        
        self.tts.speak(reply, detected_lang)
    
    def process_file(self, filepath: str):
        pretty_print("[WHISPER]", "Process audio from file.")
        if os.path.exists(filepath):
            transcript = self.stt.transcribe(filepath)
            if transcript[0]:  # Check if transcript is not empty
                self.handle_transcript(transcript)
            else:
                pretty_print("[WHISPER]", "Could not transcribe audio file")
        else:
            pretty_print("[ERROR]", f"File not found: {filepath}")
    
    def process_youtube(self, url: str):
        """Download and process YouTube audio."""
        downloaded = self.download_youtube_audio(url, OUTFILE)
        if downloaded and os.path.exists(OUTFILE):
            transcript = self.stt.transcribe(OUTFILE)
            if transcript[0]:
                self.handle_transcript(transcript)
            else:
                pretty_print("[WHISPER]", "Could not transcribe YouTube audio")
        else:
            pretty_print("[ERROR]", "Failed to download YouTube audio")
    
    def run_continuous(self):
        """Run continuous recording and processing loop."""
        try:
            while True:
                duration = self.recorder.record_once()
                if duration >= MIN_DURATION:
                    transcript = self.stt.transcribe(os.path.join(self.script_dir, OUTFILE))
                    if transcript[0]:
                        self.handle_transcript(transcript)
                    else:
                        pretty_print("[WHISPER]", "Whisper could not transcribe audio")
                else:
                    from utils import record_print
                    record_print("Recording too short, skipping transcription and response")
        except KeyboardInterrupt:
            pretty_print("[CHATBOT]", "Goodbye!")
    
    def download_youtube_audio(self, url: str, output_file: str) -> bool:
        """Download audio from YouTube video."""
        pretty_print("[YT-DLP]", f"Downloading YouTube audio: {url}")
        try:
            result = subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "wav", "--force-overwrites", "--output", output_file, url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                pretty_print("[YT-DLP]", "Download completed successfully")
                return True
            else:
                pretty_print("[YT-DLP]", f"Download failed: {result.stderr}")
                return False
        except FileNotFoundError:
            pretty_print("[YT-DLP]", "yt-dlp not found. Please install it with: pip install yt-dlp")
            return False
        except Exception as e:
            pretty_print("[YT-DLP]", f"Failed to run yt-dlp: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="AI Chatbot with voice interaction")
    parser.add_argument("--file", type=str, help="Specify audio file")
    parser.add_argument("--url", type=str, help="Specify YouTube video URL")
    parser.add_argument("--lang", type=str, default="", help="Language code to pass to whisper (e.g., zh, en, ja)")
    parser.add_argument("--prompt", type=str, default=None, help="Default prompt prefix (if not specified, uses detected language)")
    
    args = parser.parse_args()
    
    # Startup banner
    print("\n\n\n", end="")
    pretty_print("[CHATBOT]", "Start...")
    
    chatbot = ChatBot(args.lang, args.prompt)
    
    if args.url:
        chatbot.process_youtube(args.url)
    elif args.file:
        chatbot.process_file(args.file)
    else:
        chatbot.run_continuous()

if __name__ == "__main__":
    main()