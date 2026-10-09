### Text Summarizer App Using T5 Transformer

### Project Overview

The Text Summarizer App is a Natural Language Processing (NLP) application that automatically generates concise summaries from long text or dialogues using a T5 (Text-to-Text Transfer Transformer) model.

The application provides a simple web interface where users can enter text and receive a generated summary.

### Technologies Used

* Programming Language: Python
* NLP Model: T5 Transformer
* Deep Learning Framework: PyTorch
* Model Library: Hugging Face Transformers
* Dataset: SAMSum Dialogue Dataset
* Backend: FastAPI
* Frontend: HTML, CSS and JavaScript
* Server: Uvicorn
* Training Environment: Google Colab
* Development Environment: VS Code

## Features

* Automatic dialogue summarization
* Text preprocessing and tokenization
* AI-based summary generation
* FastAPI backend integration
* Custom web interface
* Local execution

### Dataset

The project uses the SAMSum dataset, which contains conversational dialogues and their corresponding human-written summaries. This dataset is suitable for training and fine-tuning models for dialogue summarization.

### Model Training

A pretrained T5 Transformer model was fine-tuned using the SAMSum dataset in Google Colab. The trained model and tokenizer files were saved and integrated into the local application.

How It Works

1. The user enters a dialogue or text into the web interface.
2. FastAPI receives the input.
3. The application preprocesses the text.
4. The T5 tokenizer converts the text into tokens.
5. The trained T5 model generates a summary using PyTorch.
6. The generated summary is returned to the frontend and displayed to the user.

## How to Run Locally

### Prerequisites

* Python installed on your computer
* Required Python libraries installed
* Trained T5 model files

### Steps

1. Download or clone this repository.

2. Ensure the trained model files are available in the "saved_summary_model" folder.

3. Install the required Python libraries if they are not already installed.

4. Open a terminal in the project directory.

5. Run the application:

   uvicorn app:app --reload

6. Open the following address in your browser:

   http://127.0.0.1:8000/

Note: The application currently runs locally and has not been deployed to a public hosting platform.

 Future Enhancements

* Support for uploaded text documents
* Multilingual text summarization
* Improved summary quality and evaluation
* Deployment as a web application

 Author

Artificial Intelligence and Data Science Engineering Student

Amrutvahini College of Engineering, Sangamner
