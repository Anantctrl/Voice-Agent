from TTS.api import TTS

# Load the model (downloads automatically the first time)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

text = "Hello! Welcome to my AI assistant.what can i do for you"

tts.tts_to_file(
    text=text,
    file_path="output.wav"
)

print("Audio generated!")

from playsound import playsound

playsound("output.wav") 