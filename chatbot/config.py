import os

# === Audio Recording Settings ===
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
CHANNELS = 1
VAD_MODE = 2  # 0: Most conservative, 3: Most sensitive
SILENCE_TIMEOUT = 1.5  # seconds
MAX_SEG_SECS = 1200.0  # Maximum recording length (seconds)
MIN_DURATION = 1.0  # Do not save if shorter than this
DEVICE_INDEX = None  # Default device
OUTFILE = "chatbot.wav"

# === Model Settings ===
# Use environment variables or default paths outside the repository
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', '/Users/hrsung/Documents/work/ai/chatbot/models/ggml-large-v3.bin')
OLLAMA_MODEL = "llama3"

# === Piper TTS Settings ===
PIPER_BIN = os.environ.get('PIPER_BIN', '/Users/hrsung/Documents/work/ai/chatbot/tools/piper/piper/build/piper')
PIPER_MODEL_EN = os.environ.get('PIPER_MODEL_EN', '/Users/hrsung/Documents/work/ai/chatbot/models/en_US-amy-medium.onnx')
PIPER_MODEL_ZH = os.environ.get('PIPER_MODEL_ZH', '/Users/hrsung/Documents/work/ai/chatbot/models/zh_CN-huayan-medium.onnx')
PIPER_SAMPLE_RATE = 22050
PIPER_ESPEAK_DATA = os.environ.get('PIPER_ESPEAK_DATA', '/Users/hrsung/.homebrew/share/espeak-ng-data')

# === Debug Settings ===
DEBUG_RECORDING = False
DEBUG_WHISPER = False
DEBUG_PIPER = False

# === Display Settings ===
PREFIX_COL = 12  # Message start column (including left and right brackets)
LINE_WIDTH = 80  # Total width
PAD = 1  # Space between prefix and message

# === Language Settings ===
DEFAULT_PROMPTS = {
    'zh': "請用中文回答，並以三句話作答",
    'en': "Please respond in English and in 3 sentences"
}