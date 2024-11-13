!pip install -U git+https://github.com/openai/whisper.git
!pip install pyannote.audio

import whisper
model = whisper.load_model("base")
print("Whisper model loaded successfully.")

import logging
import os
import json
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to your audio file and output locations
audio_path = "/content/audio.mp3"  # Adjust as needed

def transcribe_audio(file_path):
    """Transcribes the audio file using Whisper, including timestamps."""
    logging.info("Starting transcription using Whisper for file: %s", file_path)
    try:
        result = model.transcribe(file_path)
        logging.info("Transcription completed successfully.")

        # Return segments with timestamps
        segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in result["segments"]
        ]
        return segments
    except Exception as e:
        logging.error("Failed to transcribe audio: %s", e)
        return None

def perform_speaker_diarization(audio_path):
    """Performs speaker diarization using the PyAnnote pipeline with model 3.1."""
    logging.info("Starting speaker diarization on audio file: %s", audio_path)
    try:
        hf_token = "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"  
        pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

        diarization = pipeline(audio_path)

        # Save the diarization output in RTTM format for speaker info
        rttm_output_path = os.path.splitext(audio_path)[0] + ".rttm"
        with open(rttm_output_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)

        logging.info("Speaker diarization completed successfully.")
        return diarization
    except Exception as e:
        logging.error("Failed to perform speaker diarization: %s", e)
        return None

def align_transcript_with_speakers(transcript_segments, diarization):
    """Aligns Whisper transcription segments with speaker labels from PyAnnote diarization."""
    aligned_transcript = []
    for segment in transcript_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Find the corresponding speaker label from diarization
        speaker_label = "Unknown"
        for turn in diarization.itertracks(yield_label=True):
            diarization_segment, _, speaker = turn
            # Check if transcription segment overlaps with diarization segment
            if diarization_segment.intersects(Segment(start_time, end_time)):
                speaker_label = speaker
                break  # Stop searching once the correct speaker is found

        aligned_transcript.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "speaker": speaker_label
        })
    return aligned_transcript

def save_transcript(transcript, output_path="base_transcript.json"):
    """Saves the aligned transcript to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(transcript, file, ensure_ascii=False, indent=2)
        logging.info("Aligned transcript saved successfully to %s.", output_path)
        return True
    except Exception as e:
        logging.error("Failed to save transcript: %s", e)
        return False

# RUN
# Step 1: Transcribe the audio
transcript_segments = transcribe_audio(audio_path)
if not transcript_segments:
    logging.error("Transcription failed.")
else:
    # Step 2: Perform speaker diarization
    diarization = perform_speaker_diarization(audio_path)
    if not diarization:
        logging.warning("Speaker diarization was not successful.")
    else:
        # Step 3: Combine transcription with speaker labels
        aligned_transcript = align_transcript_with_speakers(transcript_segments, diarization)

        # Step 4: Save the aligned transcript with speaker labels and timestamps
        success = save_transcript(aligned_transcript)
        if success:
            logging.info("Aligned transcription with speaker labels saved successfully.")
        else:
            logging.error("Failed to save aligned transcript.")
