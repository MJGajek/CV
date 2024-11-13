!pip install espnet torchaudio speechbrain sklearn

import os
import json
import logging
import torch
import torchaudio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from espnet2.bin.asr_inference import Speech2Text
from speechbrain.pretrained import SpeakerRecognition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to your audio file
audio_path = "/content/audio.mp3"  # Adjust as needed

# Load Espnet ASR model
def load_asr_model():
    """Load the Espnet ASR model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = Speech2Text.from_pretrained("espnet/transformer_asr", device=device)
    logging.info("Espnet ASR model loaded successfully.")
    return asr_model

# Load SpeechBrain SpeakerRecognition model
def load_speaker_recognition_model():
    """Load the SpeechBrain speaker recognition model for embedding extraction."""
    speaker_recognition_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")
    logging.info("SpeechBrain speaker recognition model loaded successfully.")
    return speaker_recognition_model

# Transcribe audio with timestamps using Espnet ASR
def transcribe_audio(file_path, asr_model):
    """Transcribes audio with Espnet ASR model, including timestamps."""
    logging.info("Starting transcription with Espnet ASR for file: %s", file_path)
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        segments = asr_model(waveform[0], fs=sample_rate)
        
        # Extract transcription with timestamps
        transcript_segments = [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
            for segment in segments
        ]
        logging.info("Transcription completed successfully.")
        return transcript_segments
    except Exception as e:
        logging.error("Failed to transcribe audio: %s", e)
        return None

# Extract speaker embeddings and perform clustering for diarization
def perform_speaker_diarization(audio_path, speaker_recognition_model, transcript_segments):
    """Performs speaker diarization by clustering speaker embeddings."""
    logging.info("Starting speaker diarization on audio file: %s", audio_path)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Extract embeddings for each transcription segment
        embeddings = []
        for segment in transcript_segments:
            start, end = segment["start"], segment["end"]
            segment_waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]
            embedding = speaker_recognition_model.encode_batch(segment_waveform).detach().numpy()
            embeddings.append(embedding.squeeze())

        # Cluster embeddings to label speakers
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, linkage="ward")
        speaker_labels = clustering.fit_predict(embeddings)
        
        # Attach speaker labels to each segment
        for i, segment in enumerate(transcript_segments):
            segment["speaker"] = f"Speaker {speaker_labels[i]}"
        
        logging.info("Speaker diarization completed successfully.")
        return transcript_segments
    except Exception as e:
        logging.error("Failed to perform speaker diarization: %s", e)
        return None

# Save the aligned transcript to JSON
def save_transcript(transcript):
    """Saves the transcribed text with speaker labels to a JSON file."""
    try:
        with open("transcript_with_speakers.json", "w", encoding="utf-8") as file:
            json.dump(transcript, file, ensure_ascii=False, indent=2)
        logging.info("Transcription saved to file: transcript_with_speakers.json")
        return True
    except Exception as e:
        logging.error("Failed to save transcription: %s", e)
        return False

# RUN
# Load models
asr_model = load_asr_model()
speaker_recognition_model = load_speaker_recognition_model()

# Step 1: Transcribe the audio
transcript_segments = transcribe_audio(audio_path, asr_model)
if not transcript_segments:
    logging.error("Transcription failed.")
else:
    # Step 2: Perform speaker diarization using clustering
    diarized_transcript = perform_speaker_diarization(audio_path, speaker_recognition_model, transcript_segments)
    if not diarized_transcript:
        logging.warning("Speaker diarization was not successful.")
    else:
        # Step 3: Save the aligned transcript with speaker labels and timestamps
        success = save_transcript(diarized_transcript)
        if success:
            logging.info("Aligned subtitle transcription with speaker labels saved successfully.")
        else:
            logging.error("Failed to save aligned subtitles.")
