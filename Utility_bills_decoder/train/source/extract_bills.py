
import os
import json
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# The integration reads GOOGLE_API_KEY.
model = init_chat_model("google_genai:gemini-3.5-flash-lite")    

def extract_bill(image_path: str) -> dict:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type not in ("image/jpeg", "image/png"):
        raise ValueError(f"Unsupported image type: {path}")

    image_data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    schema = {
        "provider": "KE | LESCO | IESCO or null",
        "tariff": "string or null",
        "sanctioned_load_kw": "number or null",
        "bill_month": "YYYY-MM or null",
        "reading_date": "YYYY-MM-DD or null",
        "issue_date": "YYYY-MM-DD or null",
        "due_date": "YYYY-MM-DD or null",
        "previous_reading": "number or null",
        "current_reading": "number or null",
        "units_consumed": "number or null",
        "charges": [
            {"type": "allowed charge type", "amount": 0}
        ],
        "total_charges": "number or null",
        "taxes": [
            {"type": "allowed tax type", "amount": 0}
        ],
        "total_taxes": "number or null",
        "current_bill": "number or null",
        "arrears": "number or null",
        "payable_within_due_date": "number or null",
        "payable_after_due_date": "number or null",
    }

    prompt = f"""
    Extract the electricity bill details from this image.

    Return ONLY one valid JSON object matching this schema:
    {json.dumps(schema)}

    Rules:
    1. Include every field, even when its value is null.
    2. Use numbers, not strings, for numeric values.
    3. Never guess missing, covered, or unreadable values.
    4. Exclude names, addresses, CNICs, account,
       reference, consumer, and meter numbers.
    5. Preserve printed amounts; do not recalculate them.
    6. Convert dates to the specified ISO formats.
    7. Use only the allowed charge and tax types.
    8. Keep separate entries for repeated charge types.
    9. Record credits and subsidies as negative amounts.
    10. Use [] when there are no charge or tax entries.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{mime_type};base64,{image_data}"
                    )
                },
            },
        ]
    )

    response = model.invoke([message])
    result = response.content

    if not isinstance(result, str):
        result = "".join(
            part.get("text", "")
            for part in result
            if isinstance(part, dict)
        )

    # Handle a response wrapped in Markdown code fences.
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
        result = result.rsplit("```", 1)[0].strip()

    data = json.loads(result)

    # Basic validation: required keys must be present.
    required_fields = set(schema.keys())
    if not required_fields.issubset(data.keys()):
        missing = required_fields - set(data.keys())
        raise ValueError(f"Missing fields: {missing}")

    return data


if __name__ == "__main__":
    image_path = Path(r"Utility_bills_decoder/train/bills/IESCO_0004.jpg")

    extracted = extract_bill(image_path)

    print(json.dumps(
        extracted, indent=2, ensure_ascii=False
    ))

    Path("Utility_bills_decoder/train/train_output").mkdir(exist_ok=True) #
    Path("Utility_bills_decoder/train/train_output/IESCO_0004.json").write_text(
        json.dumps(extracted, indent=2, ensure_ascii=False), #
        encoding="utf-8",
    )

    print("\nExtraction completed.")
