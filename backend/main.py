from flask import Flask, request, jsonify, send_file
import os
import re
import json
from transformers import pipeline
from groq import Groq
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Sentiment analysis model
sentimentAnalyzer = pipeline(
    "sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Initialize Groq API
client = Groq(api_key=GROQ_API_KEY)

# Load transcript file
def loadTranscript(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Extract data from transcript
def extractCallData(transcript):
    pattern = r'\[(.*?) (\d{2}:\d{2})\]\n(.*?)(?=\[|$)'
    return re.findall(pattern, transcript, re.DOTALL)

# Analyze sentiments
def analyzeSentiment(transcript_data):
    result = []
    for data in transcript_data:
        text = data[2].strip()
        sentiment = sentimentAnalyzer(text)[0]

        result.append({
            'time': data[1],
            'text': text,
            'speaker': data[0],
            'sentiment': sentiment['label'],
            'sentiment_score': sentiment['score'],
        })
    return result

# Generating feedback
def getGroqFeedback(analysis_data):
    total_entries = len(analysis_data)
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    agent_entries = [
        entry for entry in analysis_data if entry['speaker'] == 'Sales Agent']

    for entry in analysis_data:
        sentiment_counts[entry['sentiment'].lower()] += 1

    # Prompt
    prompt = f"""Analyze the following call transcript summary and provide brief, specific feedback on the agent's performance:

    Call Summary:
    - Total exchanges: {total_entries}
    - Sentiment distribution: Positive {sentiment_counts['positive']}, Neutral {sentiment_counts['neutral']}, Negative {sentiment_counts['negative']}
    - Agent responses: {len(agent_entries)}

    Provide concise, actionable feedback in no more than 150 words."""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a concise call transcript performance analyst. Provide brief, specific feedback."},
            {"role": "user", "content": prompt}
        ],
        model="mixtral-8x7b-32768",
    )

    feedback = chat_completion.choices[0].message.content
    return feedback.strip()

# analyze the uploaded transcript
@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        transcript = loadTranscript(file_path)
        transcriptData = extractCallData(transcript)
        sentimentData = analyzeSentiment(transcriptData)

        groq_feedback = getGroqFeedback(sentimentData)

        analysis_result_path = os.path.join(
            'results', f'{filename}_analysis.json')
        with open(analysis_result_path, 'w') as result_file:
            json.dump({"analysis_data": sentimentData,
                      "groq_feedback": groq_feedback}, result_file)

        return jsonify({
            "message": "Analysis completed",
            "file": analysis_result_path,
            "feedback": groq_feedback,
            "sentiment_data": sentimentData
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Download the analysis file
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_file(os.path.join('results', secure_filename(filename)), as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=os.getenv("FLASK_DEBUG", False), port=os.getenv("FLASK_PORT", 5000))
