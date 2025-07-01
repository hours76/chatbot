import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from config import (
    SAMPLE_RATE, FRAME_SIZE, CHANNELS, VAD_MODE, SILENCE_TIMEOUT, 
    MAX_SEG_SECS, MIN_DURATION, DEVICE_INDEX, OUTFILE, DEBUG_RECORDING
)
from utils import record_print, pretty_print

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def record_once(self) -> float:
        """Record audio until silence is detected. Returns duration in seconds."""
        is_rec, buf, sil_start, seg_start, done = False, [], None, None, False
        record_print("Listening for speech...")
        speech_started = False
        recording_msg_printed = False
        
        def callback(indata, frames, *_):
            nonlocal is_rec, buf, sil_start, seg_start, done, speech_started
            pcm = indata[:, 0].tobytes()
            is_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
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
                                device=DEVICE_INDEX, callback=callback):
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
        
        sf.write(os.path.join(self.script_dir, OUTFILE), audio, SAMPLE_RATE, subtype='PCM_16')
        pretty_print("[RECORDING]", f"Saved recording as {OUTFILE} ({dur:.2f}s)")
        return dur