
import json
from urllib.parse import urlparse, parse_qs
import re
from youtube_transcript_api.formatters import TextFormatter
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
import os
from utils import filter_out_shorts_urls, delete_folder_contents


def generate_topic_transcript(query):
    results = YoutubeSearch(query, max_results=10).to_json()

    results = json.loads(results)


    delete_folder_contents('experimental_topic')

    arr_of_ids = []
    for i in range(0,10):
        arr_of_ids.append(results['videos'][i]['id'])
        

    arr_of_urls = []

    for i in range(0,10):
        url = "youtube.com" + results['videos'][i]['url_suffix']
        arr_of_urls.append(url)
        
        


    new_arr_of_urls = filter_out_shorts_urls(arr_of_urls)


    def get_transcript(url):
        url_data = urlparse(url)
        video_id = parse_qs(url_data.query)["v"][0]
        if not video_id:
            print('Video ID not found.')
            return None

        try:
            formatter = TextFormatter()

            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es'])
            text = formatter.format_transcript(transcript)
            text = re.sub('\s+', ' ', text).replace('--', '')
            return video_id, text

        except Exception as e:
            print('Error downloading transcript:', e)
            return None

    for i in range(0,len(new_arr_of_urls)):
        
     try:
        video_id, transcript_i = get_transcript(new_arr_of_urls[i])
        

        
        #write 10 different text files to experimental_topic

        if not os.path.exists('experimental_topic'):
            os.makedirs('experimental_topic')
        else:
            file_path = os.path.join("experimental_topic", f"{arr_of_ids[i]}.txt")
            with open(file_path, 'w') as file:
                file.write(transcript_i)
     except:
        pass

    
        
