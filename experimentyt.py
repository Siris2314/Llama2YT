import whisper
from pytube import YouTube
from transformers import pipeline
import os
from typing import List
import logging
import time
from tqdm import tqdm



def download_audio_from_youtube(url: str, video_name: str) -> str:
    video_url= YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename



def load_whisper_model(model_name: str = "medium"):
    return whisper.load_model(model_name)

def transcribe_audio_to_text(model, audio_path: str, language: str = "English"):
    return model.transcribe(audio_path, fp16=False, language=language)

def save_text_to_file(text: str, file_name: str):
    try:
        with open(file_name, "w+") as file:
            file.write(text)
    except (IOError, OSError, FileNotFoundError, PermissionError) as e:
        logging.debug(f"Error in file operation: {e}")

def get_text(url: str, video_name: str) -> None:
    model = load_whisper_model()
    audio_path = download_audio_from_youtube(url, video_name)
    
    with tqdm(total=100, desc="Loading Model", unit="%", ascii=False) as pbar:
        for _ in range(100):
            time.sleep(0.01)  # Simulate loading time
            pbar.update(1)

    # Show a progress bar for downloading audio
    with tqdm(total=100, desc="Downloading Audio", unit="%", ascii=False) as pbar:
        for _ in range(100):
            time.sleep(0.01)  # Simulate download time
            pbar.update(1)

    
    result = transcribe_audio_to_text(model, audio_path)
    
    with tqdm(total=100, desc="Saving Text", unit="%", ascii=False) as pbar:
        for _ in range(100):
            time.sleep(0.01)  # Simulate saving time
            pbar.update(1)
            
    save_text_to_file(result["text"], video_name + ".txt")
    
    print("Text extraction completed.")

