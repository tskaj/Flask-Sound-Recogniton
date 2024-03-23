from flask import Flask, render_template, request
from pydub import AudioSegment
import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from gtts import gTTS
import speech_recognition as sr
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads/'

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    return audio

def extract_features(audio):
    audio_array = np.array(audio.get_array_of_samples()) / (2**15)
    mfccs = librosa.feature.mfcc(y=audio_array, sr=audio.frame_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta, delta_delta])
    return features.T

def cluster_audio(features):
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(features)
    return clusters

def generate_summary(audio_text):
    sentences = sent_tokenize(audio_text)
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [sentence for sentence in sentences if sentence.lower() not in stop_words]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_sentences)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = np.sum(cosine_sim, axis=1)
    top_sentences_idx = scores.argsort()[-3:][::-1]
    summary = [filtered_sentences[i] for i in top_sentences_idx]
    return ' '.join(summary)

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data)
    return transcribed_text

def generate_summary_audio(summary_text, output_path):
    summaries_folder = 'static/summaries'
    os.makedirs(summaries_folder, exist_ok=True)
    output_file = os.path.join(summaries_folder, "summary_audio.mp3")
    tts = gTTS(text=summary_text, lang='en')
    tts.save(output_file)
    return output_file

def identify_language(audio_text):
    try:
        language = detect(audio_text)
        return language
    except:
        return "Language detection failed"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            audio = preprocess_audio(file_path)
            features = extract_features(audio)
            clusters = cluster_audio(features)
            audio_text = transcribe_audio(file_path)
            summary = generate_summary(audio_text)
            generate_summary_audio(summary, "summary_audio.mp3")
            language = identify_language(audio_text)
            return render_template('index.html', summary=summary, language=language)
    return render_template('index.html', summary=None, language=None)

if __name__ == '__main__':
    app.run(debug=True)
