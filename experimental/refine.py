#word_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)


# def get_yt_transcript(url):
#     id_input = url.split('=')[1]
#     transcript = YouTubeTranscriptApi.get_transcript(id_input)
#     formatter = TextFormatter()
#     formatted_transcript = formatter.format_transcript(transcript)
#     return formatted_transcript

# def get_token_count(text):
#     text_token_count = word_splitter.count_tokens(text=text.replace("\n", " "))
#     return text_token_count

# def create_single_doc(text):
#     return Document(page_content=text)

# def create_multiple_docs(text):
#     text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],chunk_size=3950, chunk_overlap=10)
#     docs = text_splitter.create_documents([text])
#     return docs

# def create_docs(text):
#     if get_token_count(text) > 4000:
#         return create_multiple_docs(text)
#     else:
#         return [create_single_doc(text)]
    
# text = get_yt_transcript("https://www.youtube.com/watch?v=RdYuWN3jbUo")
# docs = create_multiple_docs(text)
# #Map Reduce

# map_prompt = """
#         Write a summary of this chunk of text that includes the main points and any important details.
#         {text}
#         """
# map_prompt = PromptTemplate(template=map_prompt, input_variables=['text'])

# combine_prompt = """
#         Write a concise summary of the following text. 
#         Return your response in 2-3 sentences, which covers the key points of the text.
#         {text}
#         SUMMARY:
# """

# combine_prompt = PromptTemplate(template=combine_prompt, input_variables=['text'])

# map_reduce_chain = load_summarize_chain(llm=llm, 
#                                         chain_type='refine',
#                                         return_intermediate_steps=True,
#                                         verbose=True)

# map_reduce_outputs = map_reduce_chain({"input_documents": docs})
