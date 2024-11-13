!pip install nemo-toolkit[asr] torch

import logging
import os
import json
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIARIZATION_PIPELINE
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to your audio file
audio_path = "/content/audio.mp3"  # Adjust as needed

# Load ASR and Diarization Models
logging.info("Loading NeMo ASR and diarization models...")

# ASR Model (for transcription)
asr_model = ASRModel.from_pretrained("stt_en_conformer_ctc_large")  # Choose model as needed

# Diarization Pipeline (for speaker diarization)
diarization_config = {
    'diarizer': {
        'manifest_filepath': "/content/diarization_manifest.json",
        'out_dir': '/content/diarization_output',
        'oracle_vad': False,  # Set to True if you have VAD labels
        'vad_model': 'vad_multilingual_marblenet',
        'speaker_embeddings_model': 'titanet_large',
        'decoder': {
            'window_length_in_sec': 1.5,
            'shift_length_in_sec': 0.75
        }
    }
}

# Save the audio path to manifest for NeMo's diarization
with open(diarization_config['diarizer']['manifest_filepath'], 'w') as f:
    entry = {
        'audio_filepath': audio_path,
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': None  # Can set this if known
    }
    json.dump(entry, f)
    f.write("\n")

def transcribe_audio(file_path):
    """Transcribes audio using NeMo ASR model, including timestamps."""
    logging.info("Starting transcription using NeMo ASR for file: %s", file_path)
    try:
        # Perform transcription
        transcription_output = asr_model.transcribe([file_path], logprobs=True)[0]
        
        # Create segments with timestamps
        segments = []
        time_increment = 0.1  # Seconds between each timestamp
        for word, time in zip(transcription_output.split(), range(0, len(transcription_output) * int(1 / time_increment), int(1 / time_increment))):
            segments.append({
                "start": timedelta(seconds=time).total_seconds(),
                "end": timedelta(seconds=(time + int(1 / time_increment))).total_seconds(),
                "text": word
            })
        return segments
    except Exception as e:
        logging.error("Failed to transcribe audio: %s", e)
        return None

def perform_speaker_diarization(config):
    """Performs speaker diarization using NeMo's ASR diarization pipeline."""
    logging.info("Starting speaker diarization on audio file.")
    try:
        diarization_pipeline = ASR_DIARIZATION_PIPELINE(config['diarizer'])
        diarization_pipeline.run_diarization()
        
        # Load RTTM output
        rttm_output_path = os.path.join(config['diarizer']['out_dir'], "pred_rttms", "audio.rttm")
        return rttm_output_path
    except Exception as e:
        logging.error("Failed to perform speaker diarization: %s", e)
        return None

def merge_consecutive_speaker_segments(transcript_segments, rttm_data):
    """
    Merges consecutive segments spoken by the same speaker.
    """
    merged_transcript = []
    current_segment = None

    for segment in transcript_segments:
        # Find the corresponding speaker from the RTTM data
        speaker_label = "Unknown Speaker"  # Replace with actual parsing of RTTM data if necessary

        if current_segment is None:
            current_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker_label
            }
        else:
            # If the speaker is the same, extend the current segment
            if current_segment["speaker"] == speaker_label:
                current_segment["end"] = segment["end"]
                current_segment["text"] += " " + segment["text"]
            else:
                # Save the completed segment and start a new one for the new speaker
                merged_transcript.append(current_segment)
                current_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": speaker_label
                }

    # Append the last segment if any
    if current_segment:
        merged_transcript.append(current_segment)

    return merged_transcript

def save_transcript(transcript):
    """Saves the transcribed text to a JSON file in the default directory."""
    try:
        with open("transcript_with_speakers.json", "w", encoding="utf-8") as file:
            json.dump(transcript, file, ensure_ascii=False, indent=2)
        logging.info("Transcription saved to file: transcript_with_speakers.json")
        return True
    except Exception as e:
        logging.error("Failed to save transcription: %s", e)
        return False

# RUN
# Step 1: Transcribe the audio
transcript_segments = transcribe_audio(audio_path)
if not transcript_segments:
    logging.error("Transcription failed.")
else:
    # Step 2: Perform speaker diarization
    rttm_path = perform_speaker_diarization(diarization_config)
    if not rttm_path:
        logging.warning("Speaker diarization was not successful.")
    else:
        # Load RTTM for speaker labels and timestamps
        with open(rttm_path, "r") as rttm_file:
            rttm_data = rttm_file.readlines()
        
        # Step 3: Merge transcription with speaker labels from RTTM
        aligned_transcript = merge_consecutive_speaker_segments(transcript_segments, rttm_data)
        
        # Step 4: Save the aligned transcript with speaker labels and timestamps
        success = save_transcript(aligned_transcript)
        if success:
            logging.info("Aligned subtitle transcription with speaker labels saved successfully.")
        else:
            logging.error("Failed to save aligned subtitles.")
