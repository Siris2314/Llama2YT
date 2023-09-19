
import json
from urllib.parse import urlparse, parse_qs
import re
import os
from youtube_transcript_api.formatters import TextFormatter
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.document_loaders.youtube import YoutubeLoader

import textwrap

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<<SYS>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template

def load_splitter(chunk_size=256, chunk_overlap=20):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,length_function = len)

def yt_loader_from_url(url):
    id_input = url.split('=')[1]
    transcript = YouTubeTranscriptApi.get_transcript(id_input)
    formatter = TextFormatter()
    formatted_transcript = formatter.format_transcript(transcript)
    formatted_transcript = formatted_transcript.replace("\xa0", " ")
    text_splitter = load_splitter()
    docs = text_splitter.create_documents([formatted_transcript])
    return docs

def yt_loader_from_name(name):
    results = YoutubeSearch(name, max_results=10).to_json()

    results = json.loads(results)

    url = "youtube.com" + results['videos'][0]['url_suffix']

    url_data = urlparse(url)
    video_id = parse_qs(url_data.query)["v"][0]
    if not video_id:
            print('Video ID not found.')
            return None

    try:
        formatter = TextFormatter()
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = formatter.format_transcript(transcript)
        text = re.sub('\s+', ' ', text).replace('--', '')
        return video_id, text

    except Exception as e:
            print('Error downloading transcript:', e)
            return None
        
def delete_folder_contents(folder_path: str):
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                delete_folder_contents(item_path)
                os.rmdir(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print("\n\nSources:")
    for source in llm_response['source_documents']:
        print(source.metadata['source'])

def filter_out_shorts_urls(url_list):
    """
    Remove YouTube URLs containing 'shorts' from the list of URLs.

    Args:
        url_list (list): A list of YouTube URLs.

    Returns:
        list: A filtered list with shorts-related URLs removed.
    """
    filtered_urls = [url for url in url_list if 'shorts' not in url]
    return filtered_urls

