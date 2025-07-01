import os
import sys
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np

# ---- Suppress pkg_resources deprecation warning (Python 3.12+) ----
import warnings, sys
if sys.version_info >= (3, 12):
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources.*deprecated",
        category=UserWarning,
        module="webrtcvad",
    )
# ------------------------------------------------------------

import webrtcvad
import time
import requests
import json
import re
import argparse
from langdetect import detect

# === Parameter Settings ===
SAMPLE_RATE     = 16000
FRAME_DURATION  = 30  # ms
FRAME_SIZE      = int(SAMPLE_RATE * FRAME_DURATION / 1000)
CHANNELS        = 1
VAD_MODE        = 2     # 0: Most conservative, 3: Most sensitive
SILENCE_TIMEOUT = 1.5   # seconds
MAX_SEG_SECS    = 1200.0  # Maximum recording length (seconds)
MIN_DURATION    = 1.0   # Do not save if shorter than this
DEVICE_INDEX    = None  # Default device
OUTFILE         = "chatbot.wav"
WHISPER_MODEL   = "models/ggml-large-v3.bin"
OLLAMA_MODEL    = "llama3"
LANG_CODE = ""   # language code passed to whisper, set via --lang

DEBUG_RECORDING = False  # Recording debug message switch, silent when False
def record_print(*args, **kwargs):
    """Output only when DEBUG_RECORDING is enabled"""
    if DEBUG_RECORDING:
        print('[DEBUG_RECORDING]', *args, **kwargs)

DEBUG_WHISPER = False  # Whisper debug message switch, silent when False
def whisper_print(*args, **kwargs):
    """Output only when DEBUG_WHISPER is enabled"""
    if DEBUG_WHISPER:
        print('[DEBUG_WHISPER]', *args, **kwargs)

DEBUG_PIPER = False  # Piper debug message switch, silent when False
def piper_print(*args, **kwargs):
    if DEBUG_PIPER:
        # Always prefix with [DEBUG_PIPER]
        if args and isinstance(args[0], str) and not args[0].startswith('[DEBUG_PIPER]'):
            args = ("[DEBUG_PIPER] " + args[0],) + args[1:]
        pretty_print(*args, **kwargs)

import textwrap

PREFIX_COL  = 12    # Message start column (including left and right brackets)
LINE_WIDTH  = 80    # Existing setting: total width
PAD         = 1     # Space between prefix and message

# ---- ANSI grayscale colors ----
LIGHT_GREY = "\033[38;5;250m"   # odd lines (brighter)
DARK_GREY  = "\033[38;5;245m"   # even lines (darker)
RESET_CLR  = "\033[0m"

MSG_COUNTER = 0   # global message counter for alternating colors

def pretty_print(prefix: str, msg: str):
    """
    Print message with prefix left‑justified to PREFIX_COL,
    wrap text to LINE_WIDTH, and align continuation lines.
    All lines in the same message share the same color.
    Odd / even messages alternate between LIGHT_GREY & DARK_GREY.
    """
    global MSG_COUNTER
    color = LIGHT_GREY if MSG_COUNTER % 2 == 0 else DARK_GREY

    prefix = prefix.rjust(PREFIX_COL)
    indent = " " * (PREFIX_COL + PAD)
    wrapped = textwrap.wrap(str(msg), width=LINE_WIDTH - len(indent)) or [""]

    # first line
    print(f"{color}{prefix}{' ' * PAD}{wrapped[0]}{RESET_CLR}")

    # continuation lines
    for line in wrapped[1:]:
        print(f"{color}{indent}{line}{RESET_CLR}")

    MSG_COUNTER += 1

vad = webrtcvad.Vad(VAD_MODE)
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Piper TTS settings ===
PIPER_BIN = os.path.join(script_dir, "piper", "piper", "build", "piper")
PIPER_MODEL = os.path.join(script_dir, "models", "en_US-lessac-medium.onnx")
PIPER_SAMPLE_RATE = 22050

def record_once() -> float:
    is_rec, buf, sil_start, seg_start, done = False, [], None, None, False
    record_print("Listening for speech...")
    speech_started = False  # Flag for first speech detection
    recording_msg_printed = False

    def cb(indata, frames, *_):
        nonlocal is_rec, buf, sil_start, seg_start, done, speech_started
        pcm = indata[:, 0].tobytes()
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)
        now = time.time()

        if is_speech:
            if not is_rec:
                record_print("Speech detected, start recording...")
                is_rec = True
                buf, seg_start = [], now
                speech_started = True
            buf.append(indata.copy())
            sil_start = None
        elif is_rec:
            if sil_start is None:
                sil_start = now
                if DEBUG_RECORDING:
                    print('.', end='', flush=True)
            elif now - sil_start > SILENCE_TIMEOUT:
                if DEBUG_RECORDING:
                    print('.', end='', flush=True)
                done = True

        if is_rec and seg_start and now - seg_start > MAX_SEG_SECS:
            record_print(f"Maximum segment length {MAX_SEG_SECS}s reached, stopping recording")
            done = True

    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype='int16',
                            device=DEVICE_INDEX, callback=cb):
            while not done:
                if speech_started and not recording_msg_printed:
                    pretty_print("[RECORDING]", "Recording...")
                    recording_msg_printed = True
                sd.sleep(100)
    except Exception as e:
        pretty_print("[ERROR]", f"Recording error: {e}")
        return 0.0

    if not buf:
        pretty_print("[ERROR]", "No speech detected, nothing recorded")
        return 0.0

    audio = np.concatenate(buf, axis=0)
    dur = len(audio) / SAMPLE_RATE
    if dur < MIN_DURATION:
        record_print(f"Recording only {dur:.2f}s (< {MIN_DURATION}s), not saved")
        return 0.0

    sf.write(os.path.join(script_dir, OUTFILE), audio, SAMPLE_RATE, subtype='PCM_16')
    pretty_print("[RECORDING]", f"Saved recording as {OUTFILE} ({dur:.2f}s)")
    return dur

def run_whisper(filepath: str):
    pretty_print("[WHISPER]", "Running whisper-cpp transcription...")
    try:
        cmd = ["whisper-cpp", "--model", WHISPER_MODEL, "--file", filepath]
        if LANG_CODE:
            cmd.extend(["-l", LANG_CODE])
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # --- Show raw Whisper output (prefix each line) ---
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
        try:
            detected_lang = detect(transcript)
        except Exception:
            detected_lang = 'en'
        if detected_lang.startswith('zh'):
            detected_lang = 'zh'
        else:
            detected_lang = 'en'
        pretty_print("[LANGDETECT]", f"Whisper transcript language detected: {detected_lang}")
        return transcript, detected_lang
    except Exception as e:
        whisper_print("Whisper error:", e)
        return "", 'en'

def ask_ollama(prompt: str) -> str:
    pretty_print("[OLLAMA]", "Sending prompt to Ollama model...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True
        )
        full_reply = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "response" in data:
                    full_reply += data["response"]
                if data.get("done"):
                    break
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue
        return full_reply
    except Exception as e:
        print("Ollama call failed:", e)
        return ""

def get_piper_model_for_lang(lang_code):
    """Return the Piper model path for a given language code. Only support en and zh."""
    model_map = {
        'en': os.path.join(script_dir, "models", "en_US-lessac-medium.onnx"),
        'zh': os.path.join(script_dir, "models", "zh_CN-huayan-medium.onnx"),
    }
    # Default to English if not zh
    return model_map['zh'] if lang_code == 'zh' else model_map['en']

def speak(text: str, lang_code=None):
    lang = lang_code if lang_code in ('en', 'zh') else 'en'
    pretty_print("[PIPER]", f"Generating TTS response with Piper (lang: {lang})...")
    
    try:
        model_path = get_piper_model_for_lang(lang)
        cmd = [
            PIPER_BIN,
            "--model", model_path,
            "--espeak_data", "/Users/hrsung/.homebrew/share/espeak-ng-data",
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

def download_youtube_audio(url: str, output_file: str) -> bool:
    pretty_print("[YT-DLP]", f"Downloading YouTube audio: {url}")
    try:
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav", "--force-overwrites", "--output", output_file, url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            pretty_print("[YT-DLP]", f"yt-dlp error: {result.stderr}")
            return False
        pretty_print("[YT-DLP]", "Download complete")
        return True
    except Exception as e:
        pretty_print("[YT-DLP]", f"Failed to run yt-dlp: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Specify audio file")
    parser.add_argument("--url", type=str, help="Specify YouTube video URL")
    parser.add_argument("--lang", type=str, default="", help="Language code to pass to whisper (e.g., zh, en, ja)")
    parser.add_argument("--prompt", type=str, default=None, help="Default prompt prefix (if not specified, uses detected language)")
    args = parser.parse_args()

    LANG_CODE = args.lang.strip()

    # Startup banner
    print("\n\n\n", end="")
    pretty_print("[CHATBOT]", "Start...")

    def handle_transcript(transcript_tuple):
        transcript, detected_lang = transcript_tuple if isinstance(transcript_tuple, tuple) else (transcript_tuple, None)
        # Set prompt based on detected language if not specified
        if args.prompt:
            prompt = args.prompt
        else:
            if detected_lang == 'zh':
                prompt = f"請用中文回答，並以三句話作答"
            else:
                prompt = f"Please respond in English and in 3 sentences"
        full_prompt = f"{transcript} {prompt}".strip()
        pretty_print("[OLLAMA]", f"System prompt: {prompt}")
        reply = ask_ollama(full_prompt)
        pretty_print("[OLLAMA]", reply)
        speak(reply, detected_lang)

    if args.url:
        downloaded = download_youtube_audio(args.url, "chatbot.wav")
        if downloaded and os.path.exists("chatbot.wav"):
            transcript = run_whisper("chatbot.wav")
            if transcript:
                handle_transcript(transcript)
            else:
                pretty_print("[WHISPER]", "Whisper could not transcribe audio")
        else:
            pretty_print("[YT-DLP]", "Download failed or chatbot.wav not found")

    elif args.file:
        if not os.path.isfile(args.file):
            pretty_print("[ERROR]", f"Audio file not found: {args.file}")
            sys.exit(1)
        pretty_print("[RECORDING]", f"Using provided audio file: {args.file}")
        transcript = run_whisper(args.file)
        if transcript:
            handle_transcript(transcript)
        else:
            pretty_print("[WHISPER]", "Whisper could not transcribe audio")

    else:
        while True:
            duration = record_once()
            if duration >= MIN_DURATION:
                transcript = run_whisper(os.path.join(script_dir, OUTFILE))
                if transcript:
                    handle_transcript(transcript)
                else:
                    pretty_print("[WHISPER]", "Whisper could not transcribe audio")
            else:
                record_print("Recording too short, skipping transcription and response")
