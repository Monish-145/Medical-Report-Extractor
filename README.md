# Medical Lab Report Extraction System

A hybrid system that combines regex-based patient detail extraction with Clinical BERT for medical test result extraction from laboratory reports.

## Requirements

- Python 3.8 or higher
- pip package manager

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter NumPy compatibility issues, you may need to downgrade:

```bash
pip install "numpy<2.0"
```

## Accessing the Server

To start the server, run:
```bash
python -m uvicorn app_minimal:app --host 127.0.0.1 --port 8000
```

The server will be accessible at `http://127.0.0.1:8000`.