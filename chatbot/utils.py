import textwrap
from config import DEBUG_RECORDING, DEBUG_WHISPER, DEBUG_PIPER, PREFIX_COL, LINE_WIDTH, PAD

# ---- ANSI grayscale colors ----
LIGHT_GREY = "\033[38;5;250m"   # odd lines (brighter)
DARK_GREY = "\033[38;5;240m"    # even lines (darker)
RESET_CLR = "\033[0m"

MSG_COUNTER = 0

def pretty_print(prefix: str, message: str):
    """Print formatted messages with alternating colors and proper wrapping."""
    global MSG_COUNTER
    
    color = LIGHT_GREY if MSG_COUNTER % 2 == 0 else DARK_GREY
    indent = " " * (PREFIX_COL + PAD)
    
    wrapped = textwrap.wrap(
        message, 
        width=LINE_WIDTH - PREFIX_COL - PAD,
        subsequent_indent=""
    )
    
    if not wrapped:
        wrapped = [""]
    
    print(f"{color}{prefix:<{PREFIX_COL}}{wrapped[0]}{RESET_CLR}")
    
    for line in wrapped[1:]:
        print(f"{color}{indent}{line}{RESET_CLR}")
    
    MSG_COUNTER += 1

def record_print(*args, **kwargs):
    """Output only when DEBUG_RECORDING is enabled"""
    if DEBUG_RECORDING:
        print('[DEBUG_RECORDING]', *args, **kwargs)

def whisper_print(*args, **kwargs):
    """Output only when DEBUG_WHISPER is enabled"""
    if DEBUG_WHISPER:
        print('[DEBUG_WHISPER]', *args, **kwargs)

def piper_print(*args, **kwargs):
    """Output only when DEBUG_PIPER is enabled"""
    if DEBUG_PIPER:
        # Always prefix with [DEBUG_PIPER]
        if args and isinstance(args[0], str) and not args[0].startswith('[DEBUG_PIPER]'):
            args = ("[DEBUG_PIPER] " + args[0],) + args[1:]
        pretty_print(*args, **kwargs)