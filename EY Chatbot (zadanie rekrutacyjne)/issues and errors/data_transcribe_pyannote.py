!pip install -U git+https://github.com/openai/whisper.git
!pip install pyannote.audio

import whisper
model = whisper.load_model("base")
print("Whisper model loaded successfully.")

import logging
import os
import json
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to your audio file and output locations
audio_path = "/content/audio.mp3"  # Adjust as needed


def transcribe_audio(file_path):
    """Transcribes the audio file using Whisper, including timestamps."""
    logging.info("Starting transcription using Whisper for file: %s", file_path)
    try:
        model = whisper.load_model("base")  # Choose model size as needed
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
    """Performs speaker diarization using the PyAnnote Hugging Face pipeline."""
    logging.info("Starting speaker diarization on audio file: %s", audio_path)
    try:
        hf_token = "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"  # Replace with your actual Hugging Face token
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)

        diarization = pipeline(audio_path)

        # Save the diarization output in RTTM format for speaker info
        rttm_output_path = os.path.splitext(audio_path)[0] + ".rttm"
        with open(rttm_output_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)

        logging.info("Speaker diarization completed successfully.")
        return diarization, rttm_output_path
    except Exception as e:
        logging.error("Failed to perform speaker diarization: %s", e)
        return None, None

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
    diarization, rttm_path = perform_speaker_diarization(audio_path)
    if not diarization:
        logging.warning("Speaker diarization was not successful.")
    else:
        # Step 3: Combine transcription with speaker labels
        aligned_transcript = []
        for segment in transcript_segments:
            speaker_label = "Speaker Placeholder"  # Placeholder until diarization is parsed
            aligned_transcript.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker_label
            })

        # Step 4: Save the aligned transcript with speaker labels and timestamps
        success = save_transcript(aligned_transcript)
        if success:
            logging.info("Aligned subtitle transcription with speaker labels saved successfully.")
        else:
            logging.error("Failed to save aligned subtitles.")