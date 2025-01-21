from flask import Flask, request, jsonify
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from flask_cors import CORS
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed

# IBM Watson Speech-to-Text credentials
API_KEY = "pjY2CGxvOgEfd6CtoiP_sEcp8Q8KNzQoWHegdeTmWf-I"
URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/efebc413-bad8-4936-9a0c-8227791cae9a"

# Predefined SOS Words
SOS_WORDS = ["help", "help me", "baccho muze"]

# Setup IBM Watson Speech-to-Text
authenticator = IAMAuthenticator(API_KEY)
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(URL)

# Function to convert speech to text
def convert_speech_to_text(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/wav',
                model='en-US_BroadbandModel',
                timestamps=True,
                word_alternatives_threshold=0.9
            ).get_result()

            transcript = ''
            for result in response['results']:
                transcript += result['alternatives'][0]['transcript']
            return transcript.lower()
    except Exception as e:
        raise Exception(f"Error with Speech-to-Text: {e}")

# Function to detect SOS words
def detect_sos_condition(transcript):
    for word in SOS_WORDS:
        if word in transcript:
            return True
    return False

# Define API Endpoints
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Check if an audio file is included in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        # Save the uploaded file
        audio_file = request.files['audio']
        audio_file_path = f"temp_audio/{audio_file.filename}"
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        audio_file.save(audio_file_path)

        # Process the audio
        transcript = convert_speech_to_text(audio_file_path)
        sos_detected = detect_sos_condition(transcript)

        # Cleanup temporary file
        os.remove(audio_file_path)

        return jsonify({
            'transcript': transcript,
            'sos_detected': sos_detected
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True)
