import logging
import os
from data_utils import download_audio

# Get the parent directory of the current script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Basic settings
source = 'https://www.youtube.com/watch?v=yPIZ9b9Q-fc'
audio_output_path = os.path.join(BASE_DIR, "data", "raw", "audio")

def main(video_url):
    """Main function to download audio from a video URL."""
    logging.info("Starting the audio download process.")
    audio_path = download_audio(video_url, output_path=audio_output_path)
    if audio_path:
        logging.info("Audio downloaded and saved at: %s", audio_path)
    else:
        logging.error("Audio download failed.")

# Run the main function
if __name__ == "__main__":
    main(source)
