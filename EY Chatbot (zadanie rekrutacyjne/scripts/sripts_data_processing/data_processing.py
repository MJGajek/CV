# Install required packages
#!pip install transformers sentence_transformers faiss-gpu

import json
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and dynamic variables
TRANSCRIPT_PATH = "/Users/mjg/Desktop/EY chatbot v4/data/data_raw/base_transcript.json"  # Path to the source transcript file
MAX_TOKENS = 50  # Maximum number of words per chunk
OVERLAP_TOKENS = 10  # Number of words to repeat from the last chunk if the same speaker continues

# Filenames based on max tokens
merged_transcript_filename = f"{MAX_TOKENS}_chunk_merged_transcript.json"
faiss_index_filename = f"{MAX_TOKENS}_chunk_faiss_index.bin"
metadata_filename = f"{MAX_TOKENS}_chunk_metadata.json"

# Speaker mapping for more meaningful speaker names
speaker_mapping = {
    'SPEAKER_01': 'Grzegorz Rutkowski',
    'SPEAKER_02': 'Magda Stachura',
    'SPEAKER_03': 'Grzegorz Pietrusiński',
    'SPEAKER_04': 'Bartosz Rusek'
}

# Convert seconds to MM:SS format without fractions of seconds
def seconds_to_time_format(seconds):
    """Converts seconds to MM:SS format without fractions of seconds."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"

# Text standardization function
def standardize_text(text):
    """Standardizes text by removing excess whitespace, repetitive punctuation, and filler words."""
    # Step 1: Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 2: Consistent capitalization (capitalize first letter)
    text = text.lower()
    
    # Step 3: Remove repetitive punctuation and artifacts
    text = re.sub(r'(\.{2,}|!{2,}|\?{2,})', '.', text)
    
    # Step 4: Remove common Polish filler words or artifacts
    filler_words = ['eee', 'yyy', 'aha']
    pattern = r'\b(?:' + '|'.join(filler_words) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up any double spaces created after removing filler words
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def filter_segments(merged_transcript):
    """Removes records with empty text, NaN text, or where the speaker is 'Unknown' or 'SPEAKER_00'."""
    filtered_transcript = [
        segment for segment in merged_transcript
        if segment["text"] and segment["text"].strip() and segment["speaker"] not in ["Unknown", "SPEAKER_00"]
    ]
    logging.info("Filtered out invalid segments. Remaining segments: %d", len(filtered_transcript))
    return filtered_transcript

# Step 1: Load Transcript and Merge Consecutive Segments
def load_transcript(file_path):
    """Loads a pre-generated transcript from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            transcript = json.load(file)
        logging.info("Transcript loaded successfully.")
        return transcript
    except Exception as e:
        logging.error("Failed to load transcript: %s", e)
        return None

def merge_consecutive_segments(transcript, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    """Merges consecutive segments by the same speaker, ensuring each chunk does not exceed max_tokens.
    If the same speaker continues, each new chunk starts with the last `overlap_tokens` words from the previous chunk.
    """
    merged_transcript = []
    current_segment = {
        "start": seconds_to_time_format(transcript[0]["start"]),
        "end": seconds_to_time_format(transcript[0]["end"]),
        "text": transcript[0]["text"],
        "speaker": transcript[0]["speaker"]
    }
    current_token_count = len(current_segment["text"].split())

    for segment in transcript[1:]:
        segment_start = seconds_to_time_format(segment["start"])
        segment_end = seconds_to_time_format(segment["end"])
        segment_token_count = len(segment["text"].split())

        # Check if merging is possible based on the speaker and word limit
        if segment["speaker"] == current_segment["speaker"] and current_token_count + segment_token_count <= max_tokens:
            # Merge with the current segment
            current_segment["text"] += " " + segment["text"]
            current_segment["end"] = segment_end
            current_token_count += segment_token_count
        else:
            # Append the current segment to the merged transcript
            merged_transcript.append(current_segment)

            # Start a new segment with overlap words if the speaker is the same
            if segment["speaker"] == current_segment["speaker"]:
                last_words = current_segment["text"].split()[-overlap_tokens:]
                overlap_text = " ".join(last_words)
                current_segment = {
                    "start": segment_start,
                    "end": segment_end,
                    "text": overlap_text + " " + segment["text"],
                    "speaker": segment["speaker"]
                }
            else:
                # Start a fresh segment without overlap if the speaker changes
                current_segment = {
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment["text"],
                    "speaker": segment["speaker"]
                }

            # Update word count for the new segment
            current_token_count = len(current_segment["text"].split())

    # Append the last segment
    merged_transcript.append(current_segment)
    return merged_transcript

def save_merged_transcript(merged_transcript, output_path=merged_transcript_filename):
    """Saves the merged transcript to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(merged_transcript, file, ensure_ascii=False, indent=2)
        logging.info("Merged transcript saved successfully to %s.", output_path)
        return True
    except Exception as e:
        logging.error("Failed to save merged transcript: %s", e)
        return False

# Load, merge, and save transcript
transcript = load_transcript(TRANSCRIPT_PATH)
if transcript:
    merged_transcript = merge_consecutive_segments(transcript, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)

    # Step 2: Apply Preprocessing
    for entry in merged_transcript:
        # Map speaker names
        entry["speaker"] = speaker_mapping.get(entry["speaker"], entry["speaker"])
        # Standardize text
        entry["text"] = standardize_text(entry["text"])

    # Step 3: Filter out invalid records
    filtered_transcript = filter_segments(merged_transcript)
    save_merged_transcript(filtered_transcript)

# Step 4: Generate Embeddings and Prepare for FAISS Indexing
embedder_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Generate embeddings for each chunk in the filtered transcript
embeddings_data = []
for entry in filtered_transcript:
    embedding = embedder_model.encode(entry['text'])
    embeddings_data.append({
        'embedding': embedding,
        'start': entry['start'],
        'end': entry['end'],
        'speaker': entry['speaker'],
        'text': entry['text']
    })

# Convert embeddings to a numpy array for FAISS indexing
embeddings = np.array([item['embedding'] for item in embeddings_data])

# Step 5: Create and Save FAISS Index and Metadata
dimension = embeddings.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search

# Add embeddings to the FAISS index
index.add(embeddings)

# Save the FAISS index to a binary file
faiss.write_index(index, faiss_index_filename)
logging.info("FAISS index saved successfully to %s.", faiss_index_filename)

# Save Metadata for Retrieval
metadata = [{'start': item['start'], 'end': item['end'], 'speaker': item['speaker'], 'text': item['text']} for item in embeddings_data]
with open(metadata_filename, "w") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4)
logging.info("Metadata saved successfully to %s.", metadata_filename)

# Verification Step: Load and Check FAISS and Metadata
try:
    index = faiss.read_index(faiss_index_filename)
    logging.info("FAISS index loaded successfully.")
    logging.info("Number of embeddings in FAISS index: %d", index.ntotal)
except Exception as e:
    logging.error("Error loading FAISS index: %s", e)

try:
    with open(metadata_filename, "r") as file:
        metadata = json.load(file)
    logging.info("Metadata loaded successfully.")
    logging.info("Number of metadata entries: %d", len(metadata))
    logging.info("Sample metadata entry: %s", metadata[0])
except Exception as e:
    logging.error("Error loading metadata: %s", e)

# Consistency check
if index.ntotal == len(metadata):
    logging.info("Check passed: The number of embeddings matches the number of metadata entries.")
else:
    logging.warning("Warning: Mismatch between number of embeddings and metadata entries.")
