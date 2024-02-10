import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube
from optimum.bettertransformer import BetterTransformer


def download_audio_from_youtube(url: str, video_name: str) -> str:
    video_url= YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename

download_audio_from_youtube("https://www.youtube.com/watch?v=p0oU83Swyv8", "yes")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

whisper = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)


transcription = whisper("yes.mp3",
                        chunk_length_s=30,
                        stride_length_s=5,
                        batch_size=8)

with open('new_transcript.txt', 'w') as f:
    f.write(transcription['text'])
    