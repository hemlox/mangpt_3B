# ManGPT-3B: Local LLM Fine-Tuning on a Custom Discord Dataset

This repository documents the process of fine-tuning the `Llama-3.2-3B-uncensored` model, nicknamed **ManGPT-3B**, on a custom dataset of chat logs. The primary challenge was executing the entire workflow locally on a consumer GPU with only 8GB of VRAM (NVIDIA RTX 4060).

## My Work & Key Steps

My main contribution was engineering the data pipeline and adapting existing tools to work within my hardware constraints.

1. **Data Sourcing & Preparation:** The training data was sourced from the official Discord server for the YouTuber **manware**. Prior to data collection, an opt-out form was sent out and pinged to all server members. I wrote a custom Python script (`alpaca_formatting.py`) that:
    - Loads the raw, unstructured chat data from a `.parquet` file.
    - Processes 234,000 conversational examples.
    - Formats each example into the standard Alpaca instruction-tuning template, adding a "sliding window" of the previous 5 messages as input context.

2. **Memory-Efficient Training:** To fit the 3B model onto the 8GB GPU, I utilized the **Unsloth** library. The core training loop in `mangpt_3B.py` is an adaptation of the official Unsloth Colab notebooks, which allowed me to focus on the data engineering. The key technique was **4-bit quantization (QLoRA)** to fit the model within the VRAM constraint.

3. **Resumable Training:** The script was configured to automatically detect and resume from saved checkpoints, making the multi-hour training process more robust.

## Tech Stack

- **Model:** `chuanli11/Llama-3.2-3B-Instruct-uncensored` (an uncensored variant was chosen to reduce refusals on informal conversational data, which is the dominant style of the training corpus)
- **Core Libraries:** `PyTorch`, `Unsloth`, `HuggingFace Transformers`, `TRL`, `PEFT`, `Datasets`

## How to Run

**Prerequisites:** This project requires an NVIDIA GPU. The included dependency files are optimized for **Linux and CUDA 12.4**.

1. Clone the repository:
```bash
git clone https://github.com/hemlox/mangpt_3B.git
cd mangpt_3B
```

2. Create and activate a virtual environment (Highly Recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Note for developers: This project uses `pip-tools`. If you need to add new packages, add them to `requirements.in`, then run `pip-compile requirements.in` and `pip-sync` to update the environment).*

4. Format the dataset:
```bash
python src/alpaca_formatting.py
```

5. Begin the training process:
```bash
python src/mangpt_3B.py
```

## What I'd Do Differently

Unsloth abstracted away a lot of the training internals, convenient for getting something working, but I'd like to redo this from scratch with a raw PyTorch training loop eventually. Not anytime soon though.

## Acknowledgements

The base model used in this project is built on Meta's[Llama 3.2](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) architecture.