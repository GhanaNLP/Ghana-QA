# Ghana-QA

## Project Description

Ghana-QA is a Python script designed to automatically generate high-quality, groundable question-answer (QA) pairs from a corpus of Ghanaian news articles. The resulting dataset is ideal for training Question Answering models, enhancing search functionality, or performing SEO analysis within a Ghanaian context.

The core approach involves leveraging the **Mistral AI API** (specifically the `mistral-large-latest` model) to analyze article content and produce a set of 10 distinct, search-optimized queries and their corresponding short answers.

## Features

- **AI-Powered QA Generation:** Uses the Mistral AI model to analyze news content and generate relevant questions and answers.
- **Search-Optimized Queries:** Generates questions that mimic real-world Google search queries to ensure high practical value.
- **Checkpointing/Resuming:** Automatically resumes processing from the last saved row if an output file exists, preventing progress loss.
- **Robust Retry Logic:** Implements exponential backoff to handle API rate limits and connection errors gracefully.
- **Batch Processing:** Processes and saves articles in configurable batches for efficiency.

## Setup & Prerequisites

1. **Python 3.8+**
2. **Mistral AI API Key:** A valid API key is required. Get one from the [Mistral AI Console](https://console.mistral.ai/api-keys/).
3. **Data Format:** Your input data must be a CSV file with a column named `content` containing the raw text of the articles.

## Usage Guide

There are two ways to run this project: the simplest way using Google Colab, or the standard way on your local machine.

### Option 1: Google Colab (Recommended for ease of use)

This is the fastest way to get started and is ideal for working with large datasets without local setup.

1. **Launch the Notebook:** Click the **"Open In Colab"** badge above.
2. **Follow Instructions:** The notebook is pre-configured with cells for installation, API key input, and execution.
3. **Run Cells:** Execute the cells sequentially. You will be guided on how to mount Google Drive for persistent file saving.

### Option 2: Local Machine / Command Line Interface (CLI)

1. **Clone the Repository:**

   ```
   git clone [https://github.com/GhanaNLP/Ghana-QA.git](https://github.com/GhanaNLP/Ghana-QA.git)
   cd Ghana-QA
   ```

2. **Install Dependencies:**

   ```
   pip install pandas mistralai python-dotenv
   ```

3. **Configure Paths:** Open the Python script and update the `INPUT_PATH` and `OUTPUT_PATH` variables to point to your local CSV file locations.

4. **Run the Script:**

   ```
   python your_script_name.py
   ```

5. **Enter API Key:** The script will prompt you for your Mistral AI API key during runtime.

## Configuration

The following variables can be adjusted within the Python script to customize its behavior:

| Variable             | Description                                                  | Default Value (Needs Local Update) |
| -------------------- | ------------------------------------------------------------ | ---------------------------------- |
| `MODEL_NAME`         | The Mistral model used for generation.                       | `"mistral-large-latest"`           |
| `INPUT_PATH`         | File path to the source CSV containing news articles.        | `/media/.../myjoyonline.csv`       |
| `OUTPUT_PATH`        | File path where the resulting QA pairs are saved.            | `output_qa.csv`                    |
| `BATCH_SIZE`         | Number of articles processed and saved per API burst.        | `50`                               |
| `MAX_RETRIES`        | Maximum number of attempts for an API call on failure.       | `5`                                |
| `MAX_ARTICLE_LENGTH` | Maximum characters of an article submitted to the model (to save tokens). | `1200`                             |

## 🤝 Acknowledgements

This project is made possible by contributors and the powerful and free large language models provided by Mistral AI.
