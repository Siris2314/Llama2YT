import streamlit as st
from model_gpu import process_query, load_tokenizer_and_llm, data_loader  # Import your function from a separate file

import time




def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="https://logos-world.net/wp-content/uploads/2020/04/YouTube-Logo.png",
    )

    # Add a header with a YouTube-themed background
    st.markdown(
        """
        <h1 style="text-align: center; background-color: #ff0000; color: #ffffff; padding: 20px; border-radius: 10px">YouTube Video Summarizer</h1>
        """,
        unsafe_allow_html=True,
    )

    video_url = st.text_input("Enter the YouTube video URL:")
    time.sleep(2)

    if video_url:
        query = st.text_input("Enter your question about the video:", disabled=False)
    else:
        query = st.text_input("Enter your question about the video:", disabled=True, placeholder="Please enter a YouTube URL first")
    
    try:
        if video_url and query:
            db = data_loader(video_url)
            llm = load_tokenizer_and_llm()
            with st.spinner("Processing video and query..."):
                response = process_query(query, llm, db)

            st.header("Summary of the video:")
            st.markdown(response["answer"], unsafe_allow_html=True)

            # Sources with icons and styling
            st.subheader("Sources:")
            for source in response["sources"]:
                st.markdown(f"- <i style='font-size:20px' class='fa fa-file-text-o'></i> {source}", unsafe_allow_html=True)
    except:
        st.error("An error occurred. App stopped.")
        exit()

if __name__ == "__main__":
    main()
