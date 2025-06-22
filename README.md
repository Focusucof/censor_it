# Realtime Censor

## Overview

**Realtime Censor** is a Python application that captures live audio from your microphone, transcribes it in real time using state-of-the-art speech-to-text models, and censors (mutes) any detected swear words before the audio is played through your speakers. The application also displays a live transcription with censored words highlighted, providing both audio and visual feedback.

---

## Technologies Used

- **Python 3.10+**  
  The main programming language for the application.

- **[sounddevice](https://python-sounddevice.readthedocs.io/)**  
  For real-time audio capture from the microphone and playback to the speakers.

- **[NumPy](https://numpy.org/)**  
  For efficient audio data manipulation.

- **[RealtimeSTT](https://github.com/your-repo/RealtimeSTT)**  
  Handles real-time speech-to-text transcription, supporting models like Whisper and faster-whisper.

- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)**  
  Backend for running Whisper and faster-whisper models efficiently.

- **[better_profanity](https://github.com/snguyenthanh/better_profanity)**  
  For fast and accurate detection of swear words in transcribed text.

- **[Rich](https://github.com/Textualize/rich)**  
  For beautiful, colorized live transcription display in the terminal.

- **[colorama](https://github.com/tartley/colorama)**  
  For cross-platform colored terminal output.

---

## What It Does

1. **Captures Audio:**  
   Continuously records audio from your microphone in small chunks.

2. **Transcribes Speech:**  
   Uses a real-time local speech-to-text engine (like Whisper or faster-whisper) to transcribe each chunk of audio as you speak.

3. **Detects Swear Words:**  
   Checks each transcribed chunk for the presence of swear words using a customizable profanity filter.

4. **Censors Audio:**  
   If a swear word is detected, after a short delay, the corresponding audio chunk is muted before being played through your speakers, ensuring that offensive language is not broadcast.

---

## Usage

1. **Install dependencies:**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the application:**  
   ```
   python main.py
   ```

3. **Speak into your microphone:**  
   - Your speech will be transcribed and displayed in real time.
   - If you say a swear word, it will be muted in the audio output and highlighted in the transcription.

---

## Customization

- **Swear Words List:**  
  You can expand or modify the list of censored words in the `SWEAR_WORDS` set in `main.py`.

- **Model Selection:**  
  You can choose different Whisper or faster-whisper models for speed/accuracy trade-offs using command-line arguments.

- **Audio Devices:**  
  Specify input/output devices using `--input-device` and `--output-device` arguments.


---

## Credits

- OpenAI Whisper, faster-whisper, and CTranslate2 teams for speech-to-text technology.
- The authors of sounddevice, better_profanity, and Rich