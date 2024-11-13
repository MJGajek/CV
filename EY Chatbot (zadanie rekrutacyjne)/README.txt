# EY Chatbot Project 
# Author: Michał Gajek

## Project Overview 
This project is a chatbot solution tailored for Q&A and summarization tasks related to a Polish conference discussion. It uses a Retrieval-Augmented Generation (RAG) approach, combining retrieval and generation models to provide coherent and contextually relevant responses. The project leverages the 'eryk-mazus/polka-1.1b-chat' model and uses a FAISS index for efficient retrieval of relevant context from transcripts. 

## Features - 
**Q&A Functionality**: The chatbot can answer questions related to specific topics from the conference discussion. - **Summarization**: It provides concise summaries of the content discussed in the conference. - 
**Speaker Attribution**: Answers include information about the speaker and timestamp, allowing users to identify the source of the information. - 
**Dynamic Parameter Control**: The Gradio interface allows real-time adjustment of retrieval and generation parameters to fine-tune responses. 

## Installation To set up the environment, install the required packages listed in `requirements.txt`: ```bash pip install -r requirements.txt

/Users/mjg/Desktop/EY chatbot v4
├── EDA.ipynb					- EDA of the raw transcript data
├── Presentation.key				- presentation for the meeting
├── data						- folder contains data
│  ├── data_raw					- audio and raw transcript (first version without preprocessing)
│  └── processed_data				- pre-processed transcript and FAISS database with metadata
├── issues and errors				- record of different issues encountered during project
├── requirements.txt				- requirements for env
├── scripts						- folder with scripts
│  ├── scripts_notebooks			- because project is supposed to run in google colab it is easier to have ready scripts in notebooks to avoid copying constantly
│  └── sripts_data_processing			- scripts regarding data processing
└── zadanie_rekrutacyjne_DS_2024.docx	- task specification

Notes
Environment: Make sure to activate the correct environment (e.g., EY_Chatbot).
Dependencies: All libraries required for running the scripts are listed in requirements.txt.
Documentation: Detailed documentation is provided in the notebooks and scripts to help understand data processing, retrieval, and generation tasks.

