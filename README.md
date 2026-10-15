# AI-powered Utility Bill Decoder - WTQ_26

An AI-powered Assistant that reads Pakistani electricity bills (K-Electric, LESCO and IESCO) from images, extracts the key billing details into structured JSON, and answers customer questions about each bill in plain English.
This project was built for the WTQ Build Track challenge.

## Overview

The project runs in two levels:

| Level | Task | Output |
|-------|------|--------|
| **Level 1: Extraction** | Reads each bill and extracts its charges, taxes, readings, dates and payable amounts as a structured JSON object. Personal identifiers such as names, addresses, CNIC and account numbers are excluded. | `level1.csv` with one JSON object per bill |
| **Level 2: Explanation** | Answers a fixed set of 12 customer questions for each bill, using only the information printed on that bill. Numeric answers are calculated by code where possible. | `level2.csv` with one text answer per bill and question |

## Project Structure
```text
Maham_WTQ_26/
├── .venv/                         # Python virtual environment (not committed)
├── output/
│   ├── level1.csv                 # Level 1 results
│   └── level2.csv                 # Level 2 results
├── Utility_bills_decoder/
│   └── train/
│       ├── bills/                 # Training bill images (no answers provided)
│       ├── source/
│       │   ├── .env.example       # Template for environment variables
│       │   ├── app.py             # Application entry point
│       │   ├── extract_bills.py   # Level 1 extraction logic
│       │   └── main.py             # Pipeline: runs both levels and writes CSVs
│       ├── test/
│       │   ├── bills/             # Test bill images
│       │   └── csv/                # Level 1 and Level 2 templates
│       └── train_output/           # Output from training runs
├── .env                            # Your API key (not committed)
├── .gitignore
├── api_test.py                     # Quick check that the API key works
├── requirements.txt                # Pinned Python dependencies
└── README.md
```
---

## Setup

1. **Clone or download the project** and open the root folder in VS Code.

2. **Create and activate a virtual environment**

```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
```

4. **Configure the API key.** Create a `.env` file in the project root:

```env
   API_KEY=your-api-key-here
```
   The `.env` file is never committed. Use `source/.env.example` as a reference for the variables the project reads.

### Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `API_KEY` | Your model provider's API key | None (required) |
| `MODEL_PROVIDER` | Provider name: `google`, `anthropic` or `openai` | `google` |
| `MODEL_NAME` | Model ID from the allowed list | `gemini-3.5-flash-lite` |

The script maps `API_KEY` to `GOOGLE_API_KEY` automatically when using Google models.
### Usage
Run the pipeline from the `source` folder, pointing `--input` at the folder that contains `bills` (and optionally `csv`):

```bash
cd Utility_bills_decoder\train\source
python main.py --input ..\test --output ..\..\..\output
```

| Argument | Meaning |
|----------|---------|
| `--input` | Folder containing `bills/` and, optionally, `csv/` templates |
| `--output` | Folder where `level1.csv` and `level2.csv` are written |

Run the training bills first to check the output, then run the test bills once the templates are released.

## How It Works
1. **Load bills.** The pipeline reads every `.png` or `.jpg` in `bills/`. The bill ID is the file name without its extension, for example `KESC_0008`.
2. **Level 1.** For each bill, the image and a structured prompt are sent to the model. The model's response is parsed as JSON, then normalized in code: amounts become numbers, dates become ISO format, credits and subsidies become negative, zero lines are dropped, and the provider is taken from the file name.
3. **Level 2.** For each bill, the 12 questions and the Level 1 data are sent to the model in one request. The model returns answers keyed by question ID.
4. **Write CSVs.** Results are written to `level1.csv` and `level2.csv` in the output folder, keeping the row order and columns of the templates.
5. **Fail safely.** If a bill fails, it still gets a row with empty values, or fallback answers, so the output always has the required number of rows.

## Notes and Limitations
- Level 2 numeric answers are currently calculated by the model. Moving calculations into code would improve accuracy.
- The free Gemini tier allows 15 requests per minute. Retries with backoff handle occasional rate-limit errors.
- Results depend on the model's reading of the bill image. Bills with heavy tilt or small Urdu text may need prompt adjustments.

## AI Assistance
This project was developed with the help of AI coding assistants, **ChatGPT** (OpenAI) and **Claude** (Anthropic), which were used to design the pipeline, write and review code, and draft documentation. The project's runtime model is Gemini, as set in the configuration.
