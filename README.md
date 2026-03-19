# LLM Benchmarking

> **Note:** This documentation was generated with the assistance of AI (GitHub Copilot).

A pipeline for benchmarking large language models (LLMs) on municipal zoning code question-answering using Retrieval-Augmented Generation (RAG). The pipeline generates document embeddings from a zoning code PDF, stores them in FAISS vector stores on AWS S3, and runs multi-turn conversations against AWS Bedrock models (e.g. Meta Llama, Mistral Pixtral) and OpenAI models (e.g. GPT). Outputs are saved to S3 as CSV files for analysis.

In this pipeline, we opted to test open-source vector storage and several open-weight models while leveraging Urban Institute's existing AWS infrastructure expertise. Alternative approaches range from fully open-source stacks (e.g., Ollama for local inference) to fully managed proprietary services (e.g., OpenAI), each with different tradeoffs in cost, control, and ease of use. For our purposes, we prioritized open-source tools to maximize reproducibility and control over the pipeline.

---

## Project Structure

```
llm-benchmarking/
├── temp/                                    # Auto-created: local cache for files downloaded from S3
│   ├── Minneapolis_MN_Code_of_Ordinances.pdf  # Zoning code PDF (downloaded from S3)
│   ├── zoning-code-questions.csv              # Questions CSV (downloaded from S3)
│   └── vector_stores/                         # FAISS vector stores (downloaded/generated locally)
│       └── <model>/<splitter_type>/           # e.g. intfloat-e5-small-v2/recursive/
├── output/                                  # Auto-created: inference results before S3 upload
│   └── <model-name>/<embedding-name>/       # e.g. meta-llama/e5_large_recursive/
│       └── output_<q_type>_<date>.csv
├── scripts/
│   ├── embeddings.py                # Core functions: PDF splitting, embedding, S3 upload/download
│   ├── generate_embeddings.py       # Entry point: generate and store embeddings for a PDF
│   ├── inference.py                 # Run RAG conversations with Bedrock and OpenAI models
│   ├── prompts.py                   # System prompt definition
│   ├── run_aws_inference.ipynb      # Jupyter notebook: end-to-end inference pipeline
│   └── download-and-bind-output.py # Download and combine CSV outputs from S3
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- An AWS account with:
  - Access to **Amazon Bedrock** (with model access enabled for the models you want to benchmark)
  - An **S3 bucket** 
  - AWS credentials configured locally (via AWS CLI or environment variables)
- An **OpenAI API key** (only required if benchmarking OpenAI models)
- `poppler-utils` and `tesseract-ocr` for PDF parsing (see Step 4 below)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd llm-benchmarking
```

### 2. Create and activate a virtual environment

> **Windows users:** This codebase was developed and tested on Linux. While it can be run on Windows, we recommend using **Python 3.12** for the best compatibility with the listed dependencies. Some packages (e.g. NVIDIA CUDA libraries) are not optimized for Windows and will be skipped automatically during installation.

```bash
python3 -m venv env
source env/bin/activate          # macOS/Linux
```

```bash
py -3.12 -m venv env            # Windows (PowerShell)
env\Scripts\Activate.ps1         
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies for PDF parsing

These are required by the `unstructured` ([more info here](https://reference.langchain.com/python/langchain-unstructured/document_loaders/UnstructuredLoader)) library for high-resolution PDF partitioning:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

> On macOS, use `brew install poppler tesseract` instead.

### 5. Configure AWS credentials

If you don't have the AWS CLI installed, download and install it first by following the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

For example, on **macOS**: 
```bash
  brew install awscli
```

Then configure your credentials:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, and default region (`us-east-1`). For more details, see the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

Make sure your IAM user or role has permissions for:
- `bedrock:InvokeModel` and `bedrock:ListInferenceProfiles`
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the required bucket

### 6. Set up environment variables (OpenAI only)

If benchmarking OpenAI models, create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

### Step 1 — Upload source documents to S3

Upload the zoning code PDF and the questions CSV to S3 before running the pipeline:

```
s3://<s3-bucket-name>/Minneapolis_MN_Code_of_Ordinances.pdf
s3://<s3-bucket-name>/zoning-code-questions.csv
```

The questions CSV must have at minimum:
- A `question` column containing the question text
- An `index` column used to categorize question types (e.g. `comp1-aud1-data1`)

### Step 2 — Generate document embeddings

Run `generate_embeddings.py` to download the PDF from S3, split it into chunks, embed it using HuggingFace models, and upload the resulting FAISS vector stores back to S3:

```bash
cd scripts
python generate_embeddings.py
```

This generates embeddings for two HuggingFace models × three splitting strategies = **6 vector stores** total:

| Embedding Model | Splitting Strategy |
|---|---|
| `intfloat/multilingual-e5-large-instruct` | `recursive` |
| `intfloat/multilingual-e5-large-instruct` | `unstruct_basic` |
| `intfloat/multilingual-e5-large-instruct` | `unstruct_by_title` |
| `intfloat/e5-small-v2` | `recursive` |
| `intfloat/e5-small-v2` | `unstruct_basic` |
| `intfloat/e5-small-v2` | `unstruct_by_title` |

Vector stores are saved to S3 at `s3://<s3-bucket-name>/vector_stores/<model>/<splitter>/`.

### Step 3 — Run inference

Open and run `scripts/run_aws_inference.ipynb` in Jupyter. The notebook:

1. Loads all 6 vector stores from S3
2. Downloads the questions CSV from S3
3. Iterates over each model × embedding combination and runs multi-turn RAG conversations
4. Saves output CSV files locally and uploads them to S3 under `output/<model-name>/<embedding-name>/`

Supported models (configurable in the notebook):
- **AWS Bedrock**: `US Meta Llama 3.2 1B Instruct`, `US Mistral Pixtral Large 25.02` (via inference profiles)
- **OpenAI**: any LangChain `ChatOpenAI`-compatible model (e.g. `gpt-4o-mini`)

### Step 4 — Combine output files (optional)

After inference, use `download-and-bind-output.py` to download all output CSVs for a given model from S3 and concatenate them into a single file:

1. Update the `prefix` variable to match the model's S3 output folder (e.g. `output/mistral-ai-pixtral`)
2. Update the date filter string (`"2025-11-21"`) to match the run date
3. Run the script:

```bash
python download-and-bind-output.py
```

The combined CSV is uploaded back to S3 as `output/all_output_combined_<date>_<model>.csv`.

---

## Key Configuration

| Parameter | Location | Description |
|---|---|---|
| `S3_BUCKET_NAME` | `embeddings.py` | S3 bucket for all data |
| `embedding_models` | `generate_embeddings.py` | HuggingFace models to use for embedding |
| `splitters` | `generate_embeddings.py` | Text splitting strategies |
| `chunk_size_value` / `chunk_overlap` | `generate_embeddings.py` | Chunk size parameters |
| `question_types` | `run_aws_inference.ipynb` | Question category filters |
| `n_iter` | `run_aws_inference.ipynb` | Number of conversation iterations per question set |
| `k` | `run_aws_inference.ipynb` | Number of top context documents to retrieve per query |
| `prefix` / date filter | `download-and-bind-output.py` | Controls which S3 output files to combine |

