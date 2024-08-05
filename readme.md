# LLM API

This project provides an API for interacting with large language models (LLMs) such as Llama2 and Mistral. It allows users to select a model and send queries to the selected model, maintaining conversation context between the user and the LLM.

## Features

1. User can select a model (Llama2 or Mistral) when the program starts.
2. User can send queries to the selected model and receive answers from the LLM.
3. The program maintains conversation context between the user and the LLM, allowing for continuous interaction.
4. The project is wrapped in a Docker container for easy deployment and testing.

## Prerequisites

- Docker

## Installation

1. Clone the repository:
2. Navigate to the project directory
3. Replace the Hugging Face token:
- Open the `main.py` file.
- Locate the line where the `AutoTokenizer` is instantiated:
  ```python
  tokenizer = AutoTokenizer.from_pretrained(model_id, token="")
  ```
- Replace `""` with your actual Hugging Face token.
3. Build the Docker image

## Usage

1. Run the Docker container

2. The API will be accessible at `http://localhost:8000/query`.

3. Build the Docker image:


4. Send a POST request to the `/query` endpoint with the following JSON payload:

```json
{
  "model": "llama2",
  "question": "Your question here"
}`