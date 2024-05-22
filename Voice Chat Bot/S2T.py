from llama_index.llms.ollama import Ollama  # Our local llm
import torch
import google.generativeai as genai
import userdata
import speech_recognition as sr  # For speech to text
from gtts import gTTS  # For text to speech
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import pygame  # For playing sound
from pygame import mixer

# Initialize pygame
pygame.init()
mixer.init()

# Initialize local llm
# llm = Ollama(model="llama3", request_timeout=100.0)

# Initialize gemini
GOOGLE_API_KEY = userdata.GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel('gemini-pro')


# Function for getting audio input
def record_audio(duration=5, fs=44100):
    print("Start Speaking...")  # start recording
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wav.write('user_input.wav', fs, recording)
    print("Hearing complete...")
    return 'user_input.wav'


def get_audio_input():
    recognizer = sr.Recognizer()
    audio_file = record_audio()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Speak clearly I couldn't hear you properly, maybe try reducing your background noise")
            return "temp"
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""


def generate_response(input_text):
    #response = llm.complete(input_text) # For local llm
    chat = llm.start_chat(history=[])
    response = chat.send_message(input_text) # For Gemini
    response_text = response.text.strip()
    return response_text


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Wait for playback to finish
    pygame.mixer.music.stop()  # Stop playback
    # os.remove("response.mp3")  # Delete the file


def main():
    while True:
        user_input = get_audio_input()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        elif user_input.lower() in ["temp"]:
            print("Speak again...")
            continue
        bot_response = generate_response(user_input)
        print(f":VoiceBot {bot_response}")
        text_to_speech(bot_response)


if __name__ == "__main__":
    main()
