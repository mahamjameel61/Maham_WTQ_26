import streamlit as st
import os
import base64
import json
import re
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI-powered Pakistani Utility Bill Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Sleek UI/UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.5rem;
        color: #1E293B;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants & Model Initialization
MODEL_ID = "gemini-3.5-flash-lite"
try:
    model = init_chat_model(f"google_genai:{MODEL_ID}")
except Exception:
    model = None

FALLBACK_ANSWER = "This could not be determined from the bill."
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
CHARGE_TYPES = {"energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"}
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

LEVEL1_PROMPT = """You are extracting data from a Pakistani electricity bill image (K-Electric, LESCO or IESCO). Some bills are tilted or partly in Urdu.
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
- Use null for any value that is not printed, is blank, or unreadable. Never guess.
- Copy amounts exactly as printed. Do not recalculate or round.
- Amounts are plain numbers in PKR. Remove "Rs.", commas and labels.
- Credits and subsidies are negative numbers (-38209).
- Dates: "22nd Sep 2026" becomes "2026-09-22". "MAR 2026" becomes "2026-03".
- Do not include names, addresses, CNIC, account, reference, consumer numbers.
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
- Show key figures in PKR and double-check arithmetic.
- Write every answer in English.
- Keep each answer to 2 to 5 sentences.
Return ONLY one JSON object mapping each question_id to its answer:
{"Q1": "answer text"}
"""

# Helper functions
def image_part_from_bytes(file_bytes: bytes, suffix: str) -> dict:
    mime = MIME_TYPES.get(suffix.lower(), "image/jpeg")
    data = base64.b64encode(file_bytes).decode("ascii")
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
    if model is None:
        raise RuntimeError("Model is not initialized. Please check GOOGLE_API_KEY in your environment.")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = model.invoke([HumanMessage(content=parts)])
            return parse_json_object(text_of(response.content))
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gave up after {retries} attempts: {last_error}")

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
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d-%b-%Y", "%d-%b-%y", "%d %B %Y", "%d/%m/%Y", "%d-%m-%Y"):
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

def normalize_bill(raw: dict, bill_id: str = "bill_1") -> dict:
    prefix = bill_id.split("_")[0].upper()
    prov = {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}.get(prefix)
    out = {
        "provider": prov or raw.get("provider"),
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

# Main Application Layout
st.markdown('<div class="main-header">⚡AI-powered Pakistani Utility Bill Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract structured JSON data from bill images and get plain language answers to your billing questions.</div>', unsafe_allow_html=True)

# Session State Initialization
if "extracted_facts" not in st.session_state:
    st.session_state["extracted_facts"] = None
if "uploaded_file_bytes" not in st.session_state:
    st.session_state["uploaded_file_bytes"] = None
if "uploaded_file_suffix" not in st.session_state:
    st.session_state["uploaded_file_suffix"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Two Main Sections (Tabs)
section_1, section_2 = st.tabs(["📁 Picture & JSON Extraction", "💬 Plain Language Q&A"])

with section_1:
    st.markdown('<div class="section-title">Picture / Upload & JSON Data Extraction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.info("Upload an electricity bill image (PNG or JPG) from K-Electric, LESCO, or IESCO.")
        uploaded_file = st.file_uploader("Choose a bill image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
            st.session_state["uploaded_file_suffix"] = Path(uploaded_file.name).suffix
            st.image(uploaded_file, caption="Uploaded Bill Image", use_column_width=True)
            
            if st.button("Extract JSON Data", type="primary"):
                with st.spinner("Extracting structured data from bill..."):
                    try:
                        img_part = image_part_from_bytes(st.session_state["uploaded_file_bytes"], st.session_state["uploaded_file_suffix"])
                        raw = call_json([{"type": "text", "text": LEVEL1_PROMPT}, img_part])
                        facts = normalize_bill(raw)
                        st.session_state["extracted_facts"] = facts
                        st.success("Extraction completed successfully!")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    with col2:
        st.markdown("### Extracted JSON Data")
        if st.session_state["extracted_facts"] is not None:
            st.json(st.session_state["extracted_facts"])
        else:
            st.warning("No data extracted yet. Please upload an image and click **Extract JSON Data**.")

with section_2:
    st.markdown('<div class="section-title">Explaining Question / Answers in Plain Language</div>', unsafe_allow_html=True)
    
    if st.session_state["extracted_facts"] is None or st.session_state["uploaded_file_bytes"] is None:
        st.warning("Please upload a bill image and extract JSON data in Section 1 first before asking questions.")
    else:
        st.info("Ask any question about your electricity bill in plain English. Answers are generated directly from the bill facts.")
        
        # Display chat history
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        user_question = st.chat_input("Type your question about the bill here...")
        if user_question:
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing bill and formulating answer..."):
                    try:
                        img_part = image_part_from_bytes(st.session_state["uploaded_file_bytes"], st.session_state["uploaded_file_suffix"])
                        questions_dict = {"Q1": user_question}
                        prompt = (QA_PROMPT
                                  .replace("__FACTS__", json.dumps(st.session_state["extracted_facts"], ensure_ascii=False, indent=2))
                                  .replace("__QUESTIONS__", json.dumps(questions_dict, ensure_ascii=False, indent=2)))
                        raw_ans = call_json([{"type": "text", "text": prompt}, img_part])
                        answer_text = str(raw_ans.get("Q1", FALLBACK_ANSWER)).strip()
                    except Exception as e:
                        answer_text = f"Could not determine answer due to an error: {e}"
                    
                    st.markdown(answer_text)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})