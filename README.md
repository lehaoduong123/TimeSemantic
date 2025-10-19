# AXIS: Explainable Time Series Anomaly Detection with Large Language Models

## Overview

AXIS is an explainable time series anomaly detection framework that leverages Large Language Models to provide natural language explanations for detected anomalies.

## Project Structure

```
AXIS/
├── src/models/AXIS/
│   ├── AXIS.py                 # Main model implementation
│   ├── AXIS_test.py           # Testing framework
│   ├── dataset.py             # Dataset utilities
│   ├── Pretrain_ts_encoder.py # Time series encoder
│   └── ts_encoder_bi_bias.py  # Encoder components
├── experiments/
│   ├── configs/               # Configuration files
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # Training and testing logs
├── data/
│   └── AXIS_qa_test/          # Test dataset
├── requirements.txt           # Python dependencies
└── env_template.txt          # Environment variables template
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AXIS
```

### 2. Create conda environment and install dependencies

```bash
# Create conda environment
conda create -n AXIS python=3.11

# Activate environment
conda activate AXIS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the environment template and fill in your Hugging Face token:

```bash
cp env_template.txt .env
```

Edit `.env` file:

```bash
# Hugging Face configuration
HF_TOKEN="your_huggingface_token_here"
HF_ENDPOINT="https://hf-mirror.com"
HF_HOME="~/cache/huggingface"
TRANSFORMERS_CACHE="~/cache/huggingface/transformers"

# CUDA device configuration
CUDA_VISIBLE_DEVICES=1

# Distributed training port
DIST_PORT=12355
```

### 4. Download and extract model checkpoints

Download the pre-trained model checkpoints from Hugging Face:

```bash
huggingface-cli download thu-sail-lab/TimeSemantic checkpoints.zip --local-dir ./experiments
```

Extract the downloaded checkpoint file:

```bash
cd experiments
unzip checkpoints.zip
cd ..
```

## Usage

### Run Testing and Generate Results

1. **Set environment variables**

```bash
export HF_TOKEN="your_huggingface_token_here"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/cache/huggingface"
export TRANSFORMERS_CACHE="~/cache/huggingface/transformers"
export CUDA_VISIBLE_DEVICES=2
```

2. **Run test script**

```bash
# Set PYTHONPATH and run test
PYTHONPATH=/path/to/AXIS python src/models/AXIS/AXIS_test.py
```

Or run with one command:

```bash
# Load environment variables from .env and run
(source .env && PYTHONPATH=$(pwd) python src/models/AXIS/AXIS_test.py)
```

3. **View results**

Test results are saved in:

- **Log files**: `experiments/logs/AXIS/axis_test_YYYYMMDD_HHMMSS.txt`
- **Detailed results**: `experiments/logs/AXIS/<model_name>/results_YYYYMMDD_HHMMSS/`
  - Individual question results in YAML format (`question_XXXXXX.yaml`)
  - Test summary in `test_summary.yaml`
