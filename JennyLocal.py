from TTS.api import TTS
import pygame
import time
import sounddevice as sd
import numpy as np
import whisper
import scipy.signal
import keyboard
import os
import aiohttp
import asyncio
import json
import re
import logging

# Define global frames for audio data
frames = []

# Initialize conversation history
conversation_history = []

# Function to play audio
def play_audio(file_path):
    print(f"Playing audio file: {file_path}")
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        print(f"Audio file played successfully: {file_path}")
    except FileNotFoundError:
        print(f"Audio file not found: {file_path}")
    except Exception as e:
        print(f"Error playing audio file: {str(e)}")

# Function to record audio with * key
def record_audio_with_star_key(sample_rate=44100, channels=1):
    global frames
    frames.clear()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    print(sd.query_devices())  # List all audio devices
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        print("Press and hold the '*' key to start recording...")
        keyboard.wait('*')
        print("Recording... Release the '*' key to stop.")
        while keyboard.is_pressed('*'):
            pass
        print("Recording stopped.")

    return np.concatenate(frames, axis=0)

# Function to transcribe audio using Whisper
def transcribe_with_whisper(audio, sample_rate=16000):
    if sample_rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sample_rate)
    audio = audio.flatten()
    audio = audio.astype(np.float32)
    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")  # Debug statement
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result['text']

# Function to update conversation history
def update_conversation_history(role, content):
    global conversation_history
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 20:  # Keep only the last 20 messages
        conversation_history = conversation_history[-20:]

# Function to reset the conversation history periodically
def reset_conversation_history():
    global conversation_history
    conversation_history = []

# Async function to send prompt to Ollama API
async def send_to_ollama_api(prompt):
    async with aiohttp.ClientSession() as session:
        if not prompt:
            logging.error("Empty prompt received for Ollama API.")
            return None

        # Include conversation history in the prompt
        history_prompt = ""
        for entry in conversation_history:
            role = entry['role']
            content = entry['content']
            history_prompt += f"{role}: {content}\n"

        full_prompt = f"{history_prompt}user: {prompt}\nassistant:"

        payload = {"prompt": full_prompt, "model": "llama3:instruct"}
        headers = {"Content-Type": "application/json"}
        async with session.post("http://localhost:11434/api/generate", headers=headers, json=payload) as response:
            response_text = await response.text()
            logging.debug(f"Raw response from Ollama: {response_text}")

            try:
                fixed_response = "[" + re.sub(r'}\s*{', '},{', response_text) + "]"
                responses = json.loads(fixed_response)
                full_response = ''.join(item['response'] for item in responses if not item['done'])
                logging.info(f"Full assembled response: {full_response}")
                return full_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response from Ollama API: {str(e)}")
                return None

# Main loop
async def main():
    sd.default.device = 1  # Set the desired input device here
    sample_rate = 44100
    conversation_count = 0
    reset_interval = 10  # Reset after every 10 conversations

    while True:
        audio = record_audio_with_star_key(sample_rate=sample_rate)
        if audio.size > 0:
            transcribed_text = transcribe_with_whisper(audio, sample_rate=sample_rate)
            print(f"Transcribed Text: {transcribed_text}")
            
            update_conversation_history("user", transcribed_text)
            
            answer = await send_to_ollama_api(transcribed_text)
            if answer:
                print(f"Response: {answer}")
                update_conversation_history("assistant", answer)

                device = "cuda"
                tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False).to(device)
                tts.tts_to_file(text=answer, file_path='output.wav')

                play_audio('output.wav')

                conversation_count += 1
                if conversation_count >= reset_interval:
                    reset_conversation_history()
                    conversation_count = 0

            if transcribed_text.lower() == "goodbye":
                break

if __name__ == "__main__":
    asyncio.run(main())
