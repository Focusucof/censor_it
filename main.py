EXTENDED_LOGGING = False

# set to 0 to deactivate writing to keyboard
# try lower values like 0.002 (fast) first, take higher values like 0.05 in case it fails
WRITE_TO_KEYBOARD_INTERVAL = 0.002

# Audio censoring configuration
AUDIO_BUFFER_DURATION = 2.0  # Buffer duration in seconds to allow for transcription delay
CENSOR_BEEP_FREQUENCY = 800  # Frequency of censor beep in Hz
CENSOR_BEEP_DURATION = 0.3   # Duration of censor beep in seconds

# Common swear words list (can be expanded)
SWEAR_WORDS = {
    'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'crap', 'piss', 'dick', 'cock',
    'pussy', 'cunt', 'bastard', 'motherfucker', 'fucker', 'fucking', 'shitty',
    'asshole', 'dumbass', 'jackass', 'bullshit', 'horseshit', 'fuckin', 'fuckin\'',
    'fuck\'s', 'fuck\'re', 'fuck\'ll', 'fuck\'ve', 'fuck\'d', 'fuck\'t'
}

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Start the realtime Speech-to-Text (STT) test with various configuration options.')

    parser.add_argument('-m', '--model', type=str, # no default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, # no default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, # no default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')
    
    parser.add_argument('-d', '--root', type=str, # no default=None,
                help='Root directory where the Whisper models are downloaded to.')

    # from install_packages import check_and_install_packages
    # check_and_install_packages([
    #     {
    #         'import_name': 'rich',
    #     },
    #     {
    #         'import_name': 'pyautogui',
    #     }        
    # ])

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    console.print("System initializing, please wait")

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama
    import pyautogui
    import sounddevice as sd
    import threading
    import time
    import numpy as np
    from collections import deque
    import re

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    # Audio censoring variables
    censor_segments = []  # List of (start_time, end_time) tuples for segments to censor
    last_transcription_time = 0.0
    transcription_buffer = []  # Buffer for recent transcriptions with timing info
    buffer_start_time = time.time()

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text

    def contains_swear_words(text):
        """Check if text contains any swear words"""
        if not text:
            return False
        
        # Convert to lowercase and remove punctuation for comparison
        text_lower = re.sub(r'[^\w\s]', '', text.lower())
        words = text_lower.split()
        
        for word in words:
            if word in SWEAR_WORDS:
                return True
        return False

    def generate_censor_beep(duration, frequency, sample_rate):
        """Generate a beep sound for censoring"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(2 * np.pi * frequency * t) * 0.3  # 0.3 amplitude to avoid clipping
        return beep.astype(np.float32)

    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored, last_transcription_time, transcription_buffer, buffer_start_time

        text = preprocess_text(text)

        # Record timing for this transcription update
        current_time = time.time() - buffer_start_time
        last_transcription_time = current_time
        
        # Add to transcription buffer with timing
        if text:
            transcription_buffer.append({
                'text': text,
                'time': current_time,
                'contains_swears': contains_swear_words(text)
            })

        sentence_end_marks = ['.', '!', '?', '。'] 
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # Build Rich Text with alternating colors and censoring indicators
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")
        
        # If the current text is not a sentence-ending, display it in real-time
        if text:
            if contains_swear_words(text):
                rich_text += Text(text, style="bold red")
            else:
                rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        words = text.split()
        for i in words:
            pass

        global recorder, full_sentences, prev_text, censor_segments, transcription_buffer, buffer_start_time

        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        # Check for swear words and add to censor segments
        if contains_swear_words(text):
            current_time = time.time() - buffer_start_time
            # Estimate the time range for this text segment
            # Assume average speaking rate of 150 words per minute
            estimated_duration = len(text.split()) / 2.5  # seconds
            censor_segments.append((current_time - estimated_duration, current_time))
            console.print(f"[bold red]CENSORED: {text}[/bold red]")

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        if WRITE_TO_KEYBOARD_INTERVAL:
            pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)  # Adjust interval as needed

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        # 'input_device_index': 1,
        'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        #'on_realtime_transcription_stabilized': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 3,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt_realtime': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
    }

    args = parser.parse_args()
    if args.model is not None:
        recorder_config['model'] = args.model
        print(f"Argument 'model' set to {recorder_config['model']}")
    if args.rt_model is not None:
        recorder_config['realtime_model_type'] = args.rt_model
        print(f"Argument 'realtime_model_type' set to {recorder_config['realtime_model_type']}")
    if args.lang is not None:
        recorder_config['language'] = args.lang
        print(f"Argument 'language' set to {recorder_config['language']}")
    if args.root is not None:
        recorder_config['download_root'] = args.root
        print(f"Argument 'download_root' set to {recorder_config['download_root']}")

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024  # Number of samples per chunk
    BUFFER_SIZE = int(SAMPLE_RATE * AUDIO_BUFFER_DURATION)

    # Simple audio buffer for censoring
    audio_buffer = deque(maxlen=int(SAMPLE_RATE * AUDIO_BUFFER_DURATION / CHUNK_SIZE))

    def should_censor_audio():
        """Check if current audio should be censored based on recent transcriptions"""
        current_time = time.time() - buffer_start_time
        for start_time, end_time in censor_segments:
            if start_time <= current_time <= end_time:
                return True
        return False

    def audio_callback(indata, outdata, frames, time, status):
        # Add current audio to buffer
        audio_buffer.append(indata.copy())
        
        # Get the audio to output (with delay for censoring)
        if len(audio_buffer) > 0:
            output_audio = audio_buffer.popleft()
        else:
            output_audio = indata.copy()
        
        # Check if we should censor this audio
        if should_censor_audio():
            output_audio.fill(0)  # Replace with silence
        
        # Output the audio
        outdata[:] = output_audio

    def run_audio_stream():
        with sd.Stream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                       blocksize=CHUNK_SIZE, callback=audio_callback):
            print("Mic audio is being played through speakers with censoring. Press Ctrl+C to stop.")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("Stopped.")

    # Start audio stream in a background thread
    audio_thread = threading.Thread(target=run_audio_stream, daemon=True)
    audio_thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)

