import os
from gtts import gTTS
from groq import groq
import uuid

# Ensure the audio directory exists
AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

class VoiceTool:
    def __init__(self):
        self.client = groq(api_key=os.getenv("GROQ_API_KEY"))

    def speech_to_text(self, audio_file_path: str):
        """
        Converts speech from an audio file to text using OpenAI Whisper.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo", 
                    file=audio_file
                )
            return {"status": "success", "text": transcript.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def text_to_speech(self, text: str, lang: str = "en"):
        """
        Converts text to speech and saves it as an MP3 file using gTTS.
        Returns the path to the generated audio file.
        """
        try:
            tts = gTTS(text=text, lang=lang)
            output_filename = os.path.join(AUDIO_DIR, f"tts_{uuid.uuid4()}.mp3")
            tts.save(output_filename)
            return {"status": "success", "audio_file_path": output_filename}
        except Exception as e:
            return {"status": "error", "message": str(e)}
