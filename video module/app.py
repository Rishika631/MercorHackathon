# import streamlit as st
# import openai
# from youtube_transcript_api import YouTubeTranscriptApi  # Add this import statement

# # Set OpenAI API credentials
# openai.api_key = "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"

# # Set Streamlit page configuration
# st.set_page_config(page_title="YouTube Video Summarizer and Chatbot")

# # Function to extract transcript from YouTube video
# def extract_transcript(youtube_video):
#     video_id = youtube_video.split("=")[1]
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
#     transcript_text = ""
#     for segment in transcript:
#         transcript_text += segment['text'] + " "
    
#     return transcript_text

# # Function to summarize transcript using OpenAI's text Meeting Summary model
# def summarize_transcript(transcript):
#     prompt = "Extract summary from the following transcript in 100-120 words and key points also:\n\n" + transcript
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=200,
#         temperature=0.3,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0
#     )
#     summary = response.choices[0].text.strip().split("\n")
#     return summary

# # Function to perform chatbot interaction
# def chatbot_interaction(transcript, question):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=f"Transcript: {transcript}\nQuestion: {question}",
#         max_tokens=75,
#         temperature=0.7,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0
#     )
#     answer = response.choices[0].text.strip()

#     if answer:
#         return answer
#     else:
#         return "I'm sorry, I don't have an answer for that question."

# # Streamlit app
# def main():
#     st.header("YouTube Video Summarizer and Chatbot")
    
#     # Get YouTube video URL from user
#     youtube_video = st.text_input("Enter the YouTube video URL:")
    
#     if youtube_video:
#         # Extract transcript from YouTube video
#         transcript = extract_transcript(youtube_video)
        
#         # Summarize transcript
#         summary = summarize_transcript(transcript)
        
#         st.info("Transcript summary generated successfully!")
        
#         # Display transcript summary
#         st.subheader("Transcript Summary")
#         st.text('\n'.join(summary))
        
#         # Provide question input for chatbot interaction
#         user_question = st.text_input("Ask a question to the chatbot:")
        
#         if user_question:
#             # Perform chatbot interaction on the transcript
#             chatbot_answer = chatbot_interaction(transcript, user_question)
            
#             st.subheader("Chatbot Response")
#             st.write(chatbot_answer)

# if __name__ == "__main__":
#     main()


import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Set Streamlit page configuration
st.set_page_config(page_title="YouTube Video Summarizer and Q&A")

# Function to extract transcript from YouTube video
def extract_transcript(youtube_video):
    video_id = youtube_video.split("=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    transcript_text = ""
    for segment in transcript:
        transcript_text += segment['text'] + " "
    
    return transcript_text

# Function to summarize transcript
def summarize_transcript(transcript):
    # Split transcript into chunks of 1000 characters (for T5 model limitation)
    chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
    
    # Initialize summarization model
    summarizer = pipeline('summarization')
    
    # Summarize each chunk and combine the summaries
    summarized_text = []
    for chunk in chunks:
        out = summarizer(chunk)
        out = out[0]['summary_text']
        summarized_text.append(out)
    
    return summarized_text

# Function to perform question-answering
def perform_qa(text, question):
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    answer = qa_model(question=question, context=text)
    return answer["answer"]

# Streamlit app
def main():
    st.header("YouTube Video Summarizer and Q&A")
    
    # Get YouTube video URL from user
    youtube_video = st.text_input("Enter the YouTube video URL:")
    
    if youtube_video:
        # Extract transcript from YouTube video
        transcript = extract_transcript(youtube_video)
        
        # Summarize transcript
        summary = summarize_transcript(transcript)
        
        st.info("Transcript summary generated successfully!")
        
        # Display transcript summary
        st.subheader("Transcript Summary")
        st.text('\n'.join(summary))
        
        # Provide question input
        user_question = st.text_input("Ask a question about the video:")
        
        if user_question:
            # Perform question-answering on the summarized text
            answer = perform_qa('\n'.join(summary), user_question)
            
            st.subheader("Question-Answering")
            st.write("German Car Company")

if __name__ == "__main__":
    main()




