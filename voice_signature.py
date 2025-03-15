from flask import Flask, request, jsonify
import os
import random
import librosa
import numpy as np
import assemblyai as aai
import soundfile as sf
import noisereduce as nr
import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load SpeechBrain speaker verification model
spkrec_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_spkrec"
)

# AssemblyAI API Key (Use environment variable for security)
API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "d5a05d4271894a61ace9741605c8a7e8")
aai.settings.api_key = API_KEY

# Predefined sentences for training
SENTENCES = [
    "Technology is evolving every single day.",
    "The weather today is quite unpredictable.",
    "Artificial intelligence is shaping the future.",
    "Communication is key to building relationships.",
    "A healthy lifestyle requires balance and discipline.",
    "Reading books can expand your knowledge and creativity.",
    "Practice makes progress, not necessarily perfection.",
    "Traveling to new places broadens your perspective.",
    "Learning a new language takes time and dedication.",
    "Every challenge is an opportunity to grow stronger."
]

# Temporary storage for voice signatures and sessions
voice_signatures = {}
user_sessions = {}

def get_random_sentences(n=3):
    return random.sample(SENTENCES, n)

def noise_reduction(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    sf.write(audio_file, reduced_noise, sr)
    return audio_file

def transcribe_audio(audio_file, max_retries=3):
    transcriber = aai.Transcriber()
    for attempt in range(max_retries):
        try:
            transcript = transcriber.transcribe(audio_file)
            if transcript.status == "completed":
                return transcript.text.strip().lower()
            return None
        except Exception as e:
            print(f"Error: {e}. Retrying ({attempt + 1}/{max_retries})...")
    return None

def is_exact_match(transcribed_text, expected_text):
    return transcribed_text == expected_text.lower()

def extract_voice_features(audio_file):
    signal, _ = torchaudio.load(audio_file)
    return spkrec_model.encode_batch(signal).squeeze(0).detach().numpy()

def compare_voice_signatures(sig1, sig2, threshold=0.85):
    sig1 = sig1.flatten()
    sig2 = sig2.flatten()
    similarity_score = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
    return similarity_score > threshold

@app.route('/api/start_training', methods=['POST'])
def start_training():
    session_id = str(random.randint(1000, 9999))
    selected_sentences = get_random_sentences()
    user_sessions[session_id] = {'sentences': selected_sentences, 'signatures': []}
    return jsonify({'session_id': session_id, 'sentences': selected_sentences})

@app.route('/api/verify_training', methods=['POST'])
def verify_training():
    session_id = request.form.get('session_id')
    if session_id not in user_sessions:
        return jsonify({'error': 'Invalid session ID'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']
    sentence = request.form.get('sentence')
    if sentence not in user_sessions[session_id]['sentences']:
        return jsonify({'error': 'Invalid sentence'}), 400

    filename = secure_filename(f"{session_id}_{sentence.replace(' ', '_')}.wav")
    audio_file.save(filename)

    noise_reduction(filename)
    transcribed_text = transcribe_audio(filename)
    if not transcribed_text or not is_exact_match(transcribed_text, sentence):
        os.remove(filename)
        return jsonify({'success': False, 'error': 'Verification failed', 'transcribed': transcribed_text}), 400

    signature = extract_voice_features(filename)
    user_sessions[session_id]['signatures'].append(signature)
    os.remove(filename)
    return jsonify({'success': True, 'transcribed': transcribed_text})

@app.route('/api/create_signature', methods=['POST'])
def create_signature():
    session_id = request.form.get('session_id')
    if session_id not in user_sessions or not user_sessions[session_id]['signatures']:
        return jsonify({'error': 'Training not completed'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']
    filename = secure_filename(f"{session_id}_signature.wav")
    audio_file.save(filename)

    noise_reduction(filename)
    transcribed_signature = transcribe_audio(filename)
    if not transcribed_signature:
        os.remove(filename)
        return jsonify({'error': 'Could not transcribe signature'}), 400

    user_signature = extract_voice_features(filename)
    avg_voice_signature = np.mean(user_sessions[session_id]['signatures'], axis=0)
    voice_signatures[session_id] = {
        'signature': user_signature,
        'text': transcribed_signature,
        'avg_training': avg_voice_signature
    }
    os.remove(filename)
    return jsonify({'success': True, 'signature_text': transcribed_signature})

@app.route('/api/verify_signature', methods=['POST'])
def verify_signature():
    session_id = request.form.get('session_id')
    if session_id not in voice_signatures:
        return jsonify({'error': 'No signature found for this session'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']
    filename = secure_filename(f"{session_id}_verify.wav")
    audio_file.save(filename)

    noise_reduction(filename)
    verification_text = transcribe_audio(filename)
    if not verification_text:
        os.remove(filename)
        return jsonify({'error': 'Could not transcribe verification'}), 400

    stored_signature = voice_signatures[session_id]
    if not is_exact_match(verification_text, stored_signature['text']):
        os.remove(filename)
        return jsonify({'success': False, 'error': 'Text mismatch', 'transcribed': verification_text}), 400

    test_signature = extract_voice_features(filename)
    match = compare_voice_signatures(stored_signature['signature'], test_signature)
    os.remove(filename)
    return jsonify({'success': match, 'transcribed': verification_text})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)