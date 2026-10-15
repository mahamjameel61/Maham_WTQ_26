import argparse
import base64
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# The Google integration reads GOOGLE_API_KEY. Accept API_KEY from .env as well.
model = init_chat_model("google_genai:gemini-3.5-flash-lite")   

MODEL_ID = "gemini-3.5-flash-lite"
model = init_chat_model(f"google_genai:{MODEL_ID}")

FALLBACK_ANSWER = "This could not be determined from the bill."
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
CHARGE_TYPES = {"energy", "fixed", "fpa", "quarterly_adjustment",
                "surcharge", "meter_rent", "subsidy", "other"}
TAX_TYPES = {"gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"}
LEVEL1_FIELDS = [
    "provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date",
    "issue_date", "due_date", "previous_reading", "current_reading",
    "units_consumed", "charges", "total_charges", "taxes", "total_taxes",
    "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date",
]
NUMBER_FIELDS = [
    "sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed",
    "total_charges", "total_taxes", "current_bill", "arrears",
    "payable_within_due_date", "payable_after_due_date",
]

LEVEL1_PROMPT = """You are extracting data from a Pakistani electricity bill image
(K-Electric, LESCO or IESCO). Some bills are tilted or partly in Urdu.

Return ONLY one JSON object, with no markdown and no commentary, using exactly these keys:
{
  "provider": "KE" or "LESCO" or "IESCO",
  "tariff": "string, exactly as printed, e.g. A1-R",
  "sanctioned_load_kw": number,
  "bill_month": "YYYY-MM",
  "reading_date": "YYYY-MM-DD",
  "issue_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "previous_reading": number,
  "current_reading": number,
  "units_consumed": number,
  "charges": [{"type": "energy|fixed|fpa|quarterly_adjustment|surcharge|meter_rent|subsidy|other", "amount": number}],
  "total_charges": number,
  "taxes": [{"type": "gst|electricity_duty|income_tax|municipal_tax|other_tax", "amount": number}],
  "total_taxes": number,
  "current_bill": number,
  "arrears": number,
  "payable_within_due_date": number,
  "payable_after_due_date": number
}

Rules:
- Use null for any value that is not printed, is blank, is covered by a grey box, or is unreadable. Never guess.
- Copy amounts exactly as printed. Do not recalculate or round.
- Amounts are plain numbers in PKR. Remove "Rs.", commas and labels.
- Credits and subsidies are negative numbers (for example, 38209CR becomes -38209).
- Dates: "22nd Sep 2026" becomes "2026-09-22". "MAR 2026" becomes "2026-03" for bill_month.
- charges: one entry per line in the charges section. Skip lines printed as 0. Keep repeated types as separate entries.
- taxes: classify each line by the section it appears in, not by its name. If only a combined tax amount is printed, use "taxes": [] and fill total_taxes.
- payable_after_due_date: if several late-payment amounts are shown, use the highest.
- due_date is the last date to pay without a late payment surcharge.
- Do not include names, addresses, CNIC, account, reference, consumer, contract or meter numbers.
- Use [] when a list has no entries.
"""

QA_PROMPT = """You answer customer questions about one Pakistani electricity bill.
The bill image is attached. The extracted bill data is below.

Bill data (JSON):
__FACTS__

Questions (JSON, keyed by question_id):
__QUESTIONS__

Rules:
- Answer ONLY from the bill. Do not use outside tariff or rate knowledge.
- If the bill does not show a value you need, say so plainly. Do not guess.
- Show the key figures in PKR and double-check the arithmetic.
- If you estimate something, say "This is an estimate" and explain how you calculated it.
- If a question has more than one reasonable meaning, state the interpretation you used.
- Do not include names, addresses, CNIC, account, reference, consumer, contract or meter numbers.
- Write every answer in English, even when the bill contains Urdu.
- Keep each answer to 2 to 5 sentences.

Return ONLY one JSON object that maps each question_id to its answer, for example:
{"Q1": "answer text", "Q2": "answer text"}
"""


# ---------- Model helpers ----------

def image_part(path: Path) -> dict:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image type: {path.name}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the model response")
    return json.loads(match.group(0))


def call_json(parts: list, retries: int = 3) -> dict:
    """Call the model and parse a JSON object, retrying on errors or bad output."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = model.invoke([HumanMessage(content=parts)])
            return parse_json_object(text_of(response.content))
        except Exception as exc:
            last_error = exc
            print(f"    attempt {attempt} failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gave up after {retries} attempts: {last_error}")


# ---------- Normalization (deterministic, runs after the model) ----------

def provider_from_id(bill_id: str):
    prefix = bill_id.split("_")[0].upper()
    return {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}.get(prefix)


def to_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).upper().replace(",", "").replace("RS.", "").replace("RS", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group())
    if "CR" in text and number > 0:
        number = -number
    return int(number) if number.is_integer() else number


def to_date(value):
    if not value:
        return None
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", str(value).strip(), flags=re.I)
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d-%b-%Y", "%d-%b-%y",
                "%d %B %Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def to_month(value):
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m", "%b %Y", "%b-%y", "%b-%Y", "%B %Y", "%b %y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    full_date = to_date(text)
    return full_date[:7] if full_date else None


def normalize_lines(lines, allowed: set, default_type: str) -> list:
    result = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        amount = to_number(line.get("amount"))
        if amount in (None, 0):
            continue
        kind = str(line.get("type", "")).strip().lower()
        if kind not in allowed:
            kind = default_type
        if kind == "subsidy" and amount > 0:
            amount = -amount
        result.append({"type": kind, "amount": amount})
    return result


def normalize_bill(raw: dict, bill_id: str) -> dict:
    out = {
        "provider": provider_from_id(bill_id) or raw.get("provider"),
        "tariff": raw.get("tariff") or None,
        "bill_month": to_month(raw.get("bill_month")),
        "reading_date": to_date(raw.get("reading_date")),
        "issue_date": to_date(raw.get("issue_date")),
        "due_date": to_date(raw.get("due_date")),
        "charges": normalize_lines(raw.get("charges"), CHARGE_TYPES, "other"),
        "taxes": normalize_lines(raw.get("taxes"), TAX_TYPES, "other_tax"),
    }
    for field in NUMBER_FIELDS:
        out[field] = to_number(raw.get(field))
    if out["bill_month"] is None and out["issue_date"]:
        out["bill_month"] = out["issue_date"][:7]
    return {field: out.get(field) for field in LEVEL1_FIELDS}


def empty_bill(bill_id: str) -> dict:
    result = {field: None for field in LEVEL1_FIELDS}
    result["provider"] = provider_from_id(bill_id)
    result["charges"] = []
    result["taxes"] = []
    return result


# ---------- Level 1 and Level 2 ----------

def extract_level1(image_path: Path, bill_id: str) -> dict:
    raw = call_json([{"type": "text", "text": LEVEL1_PROMPT}, image_part(image_path)])
    return normalize_bill(raw, bill_id)


def answer_questions(image_path: Path, facts: dict, questions: dict) -> dict:
    prompt = (QA_PROMPT
              .replace("__FACTS__", json.dumps(facts, ensure_ascii=False, indent=2))
              .replace("__QUESTIONS__", json.dumps(questions, ensure_ascii=False, indent=2)))
    raw = call_json([{"type": "text", "text": prompt}, image_part(image_path)])
    answers = {}
    for qid in questions:
        value = raw.get(qid)
        answers[qid] = str(value).strip() if value else FALLBACK_ANSWER
    return answers


# ---------- CSV helpers ----------

def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames


def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


# ---------- Main pipeline ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Utility Bill Decoder")
    parser.add_argument("--input", default=r"Utility_bills_decoder\train\test", #
                        help="Folder containing bills/ and optionally csv/")
    parser.add_argument("--output", default="output", help="Folder for the CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    bills_dir = input_dir / "bills"
    csv_dir = input_dir / "csv"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in bills_dir.iterdir() if p.suffix.lower() in MIME_TYPES}
    if not images:
        raise SystemExit(f"No .png or .jpg bills found in {bills_dir}")

    # ----- Level 1 -----
    l1_template = csv_dir / "level1.csv"
    if l1_template.exists():
        l1_rows, _ = read_csv(l1_template)
        bill_ids = [row["bill_id"] for row in l1_rows]
    else:
        print("level1.csv template not found. Building bill list from images.")
        bill_ids = sorted(images)

    facts_by_bill = {}
    level1_out = []
    for bill_id in bill_ids:
        print(f"[Level 1] {bill_id}")
        try:
            if bill_id not in images:
                raise FileNotFoundError(f"No image found for {bill_id}")
            facts = extract_level1(images[bill_id], bill_id)
        except Exception as exc:
            print(f"  failed, writing empty record: {exc}")
            facts = empty_bill(bill_id)
        facts_by_bill[bill_id] = facts
        level1_out.append({
            "bill_id": bill_id,
            "json": json.dumps(facts, ensure_ascii=False, separators=(",", ":")),
        })

    write_csv(out_dir / "level1.csv", ["bill_id", "json"], level1_out)
    print(f"Wrote {out_dir / 'level1.csv'} ({len(level1_out)} rows)")

    # ----- Level 2 -----
    l2_template = csv_dir / "level2.csv"
    if not l2_template.exists():
        print("level2.csv template not found. Skipping Level 2 until the questions are released.")
        return

    l2_rows, l2_fields = read_csv(l2_template)
    questions_by_bill = {}
    for row in l2_rows:
        questions_by_bill.setdefault(row["bill_id"], {})[row["question_id"]] = row["question"]

    answers_by_bill = {}
    for bill_id, questions in questions_by_bill.items():
        print(f"[Level 2] {bill_id} ({len(questions)} questions)")
        try:
            if bill_id not in images:
                raise FileNotFoundError(f"No image found for {bill_id}")
            facts = facts_by_bill.get(bill_id, empty_bill(bill_id))
            answers_by_bill[bill_id] = answer_questions(images[bill_id], facts, questions)
        except Exception as exc:
            print(f"  failed, using fallback answers: {exc}")
            answers_by_bill[bill_id] = {qid: FALLBACK_ANSWER for qid in questions}

    for row in l2_rows:
        row["answer"] = answers_by_bill.get(row["bill_id"], {}).get(
            row["question_id"], FALLBACK_ANSWER)

    write_csv(out_dir / "level2.csv", l2_fields, l2_rows)
    print(f"Wrote {out_dir / 'level2.csv'} ({len(l2_rows)} rows)")


if __name__ == "__main__":
    main()