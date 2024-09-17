import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

# Flask backend URl
API_URL = os.getenv("API_URL")

# Upload a call transcript file for analysis
def upload_file(file):
    files = {'file': file}
    response = requests.post(f"{API_URL}/analyze", files=files)
    return response.json()

# Download the analysis file
def download_analysis(filename):
    download_url = f"{API_URL}/download/{filename}"
    response = requests.get(download_url)
    return response.content

# Bar chart for sentiment distribution
def plot_sentiment_distribution(sentiment_data):
    df = pd.DataFrame(sentiment_data)
    sentiment_counts = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'blue', 'red'])
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Pie chart for sentiment scores
def plot_sentiment_pie_chart(sentiment_data):
    df = pd.DataFrame(sentiment_data)
    sentiments = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sentiments, labels=sentiments.index, autopct='%1.1f%%',
           startangle=90, colors=['green', 'blue', 'red'])
    ax.axis('equal')
    st.pyplot(fig)


# Header
st.title("Call Transcript Sentiment Analysis")
st.write("Upload a transcript file to analyze sentiments and receive feedback on agent performance.")

# File upload
uploaded_file = st.file_uploader(
    "Choose a transcript file", type="txt", accept_multiple_files=False)

if uploaded_file is not None:
    with st.spinner('Analyzing transcript...'):
        result = upload_file(uploaded_file)

        if 'sentiment_data' in result:
            sentiment_data = result['sentiment_data']

            # Feedback on agent performance
            if 'feedback' in result:
                st.subheader("Agent Performance Feedback")
                st.write(result['feedback'])

            # Sentiment analysis table
            st.subheader("Sentiment Analysis Results")
            df = pd.DataFrame(sentiment_data)
            st.dataframe(df)

            # Bar chart
            st.subheader("Sentiment Distribution (Bar Chart)")
            plot_sentiment_distribution(sentiment_data)

            # Pie chart
            st.subheader("Sentiment Breakdown (Pie Chart)")
            plot_sentiment_pie_chart(sentiment_data)

            # Download analysis file
            st.subheader("Download Analysis")
            analysis_file = f"{uploaded_file.name}_analysis.json"

            if st.button("Generate Analysis File"):
                analysis_content = download_analysis(analysis_file)
                st.download_button(
                    label="Download Analysis File",
                    data=analysis_content,
                    file_name=analysis_file,
                    mime="application/json"
                )
        else:
            st.error(
                "Error: Something went wrong...")
