EXTENDED_LOGGING = False

# set to 0 to deactivate writing to keyboard
# try lower values like 0.002 (fast) first, take higher values like 0.05 in case it fails
WRITE_TO_KEYBOARD_INTERVAL = 0.002

# Audio censoring configuration
AUDIO_BUFFER_DURATION = 2.8  # Increased buffer for better censoring accuracy at the cost of a slight delay
CENSOR_BEEP_FREQUENCY = 1000  # Frequency of censor beep in Hz
CENSOR_BEEP_DURATION = 0.1   # Duration of censor beep in seconds

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

    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit.')

    parser.add_argument('-m', '--model', type=str, # no default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, # no default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, # no default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')
    
    parser.add_argument('-d', '--root', type=str, # no default=None,
                help='Root directory where the Whisper models are downloaded to.')

    parser.add_argument('--input-device', type=int, default=None,
                        help='Index of the input device to use for transcription.')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Index of the output device to use for audio playback.')

    args = parser.parse_args()

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

    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        sys.exit(0)

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
    processed_for_censoring_text = "" # Tracks text processed for censoring

    end_of_sentence_detection_pause = 0.2
    unknown_sentence_detection_pause = 0.5
    mid_sentence_detection_pause = 0.5

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

    def censor_swear_words_in_text(text):
        """Replaces swear words in text with asterisks."""
        pattern = r"\b[\w']+\b"
        
        def replace_swear(match):
            word = match.group(0)
            if word.lower() in SWEAR_WORDS:
                return '*' * len(word)
            return word
            
        return re.sub(pattern, replace_swear, text)

    def contains_swear_words(text):
        """Check if text contains any swear words"""
        if not text:
            return False
        
        # Fixed to correctly tokenize words, including those with apostrophes.
        words = re.findall(r"\b[\w']+\b", text.lower())
        
        for word in words:
            if word in SWEAR_WORDS:
                return True
        return False

    def generate_censor_beep(duration, frequency, sample_rate):
        """Generate a beep sound for censoring"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(2 * np.pi * frequency * t) * 0.005  # 0.3 amplitude to avoid clipping
        return beep.astype(np.float32)

    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored, last_transcription_time, transcription_buffer, buffer_start_time, censor_segments, processed_for_censoring_text

        # CENSORING LOGIC MOVED HERE FOR REAL-TIME RESPONSE
        # Determine the new text that has been transcribed since the last call
        new_text_fragment = text
        if text.startswith(processed_for_censoring_text):
            new_text_fragment = text[len(processed_for_censoring_text):]

        if new_text_fragment and contains_swear_words(new_text_fragment):
            current_time = time.time() - buffer_start_time
            # Estimate duration of the new text fragment.
            estimated_duration = len(new_text_fragment.split()) / 2.5
            text_start_time = current_time - estimated_duration

            # Censor the time range where the new fragment was likely spoken
            # Add a small buffer on both ends to be safe
            censor_segments.append((text_start_time - 0.3, current_time + 0.3))

        processed_for_censoring_text = text # Update the tracker

        # VISUAL DISPLAY LOGIC
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
            censored_sentence = censor_swear_words_in_text(sentence)
            if i % 2 == 0:
                rich_text += Text(censored_sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(censored_sentence, style="cyan") + Text(" ")
        
        # If the current text is not a sentence-ending, display it in real-time
        if text:
            censored_text = censor_swear_words_in_text(text)
            rich_text += Text(censored_text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text

        # This function now only handles finalized sentences for keyboard output
        # and maintaining the list of full sentences.
        # Real-time censoring is handled in `text_detected`.

        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        full_sentences.append(text)
        prev_text = ""
        text_detected("") # Update display to show the sentence has been finalized

        if WRITE_TO_KEYBOARD_INTERVAL:
            censored_text = censor_swear_words_in_text(text)
            pyautogui.write(f"{censored_text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'tiny.en',
        'device': 'cuda',
        'compute_type': 'auto', # Changed from float16 to auto to let the library decide the best compute type for the hardware.
        'download_root': None,
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.3, # Lowered for higher sensitivity to speech
        'webrtc_sensitivity': 1,
        'post_speech_silence_duration': 0.01, # Lowered for faster response
        'min_length_of_recording': 0.02, # Lowered for faster response
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': False,
        'early_transcription_on_silence': 0.1, # Lowered for faster transcription on silence
        'beam_size': 1, # Set to 1 for greedy decoding, which is the fastest.
        'beam_size_realtime': 1, # Set to 1 for greedy decoding, which is the fastest for real-time.
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
    if args.input_device is not None:
        recorder_config['input_device_index'] = args.input_device
        print(f"Argument 'input_device' set to {recorder_config['input_device_index']}")

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
        """Check if the audio about to be played should be censored."""
        # The audio being played was recorded AUDIO_BUFFER_DURATION seconds ago.
        # We calculate the time range for the audio chunk that is about to be played.
        playback_time = (time.time() - buffer_start_time) - AUDIO_BUFFER_DURATION
        chunk_duration = CHUNK_SIZE / SAMPLE_RATE
        
        # Check if the time interval of this playback chunk overlaps with any censor segment.
        for start_time, end_time in censor_segments:
            # If the playback interval [playback_time, playback_time + chunk_duration]
            # overlaps with the censor interval [start_time, end_time], then censor.
            if max(start_time, playback_time) < min(end_time, playback_time + chunk_duration):
                return True
        return False

    def audio_callback(indata, outdata, frames, time, status):
        # Add current audio to the end of the buffer
        audio_buffer.append(indata.copy())
        
        # If the buffer isn't full yet, play silence. This creates the delay needed for transcription.
        if len(audio_buffer) < audio_buffer.maxlen:
            output_audio = np.zeros_like(indata)
        else:
            # Once the buffer is full, start playing the delayed audio from the start of the buffer.
            output_audio = audio_buffer.popleft()
        
        # Check if the audio we are about to play should be censored
        if should_censor_audio():
            beep_duration = frames / SAMPLE_RATE
            beep = generate_censor_beep(beep_duration, CENSOR_BEEP_FREQUENCY, SAMPLE_RATE)
            beep_int16 = (beep * 32767).astype(np.int16)
            # Reshape to match the expected output shape (frames, channels)
            output_audio = beep_int16.reshape(-1, 1)
        
        # Output the audio
        outdata[:] = output_audio

    def run_audio_stream():
        with sd.Stream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                       blocksize=CHUNK_SIZE, callback=audio_callback,
                       device=(args.input_device, args.output_device)):
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

