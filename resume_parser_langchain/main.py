import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


def to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/png"
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def parse_resume(image_path: Path, model_name: str) -> ResumeData:
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(ResumeData)
    data_url = to_data_url(image_path)

    messages = [
        SystemMessage(
            content=(
                "Extract resume fields from this image. "
                "Return null or empty lists when missing."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Parse into the provided schema.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ]
        ),
    ]

    return structured_llm.invoke(messages)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Parse resume image to structured JSON.")
    parser.add_argument("--input", required=True, help="Path to resume image file.")
    parser.add_argument(
        "--output",
        default="resume_data.json",
        help="Path to output JSON file (default: resume_data.json).",
    )
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME", "gpt-4.1")
    image_path = Path(args.input)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Input file not found: {image_path}")

    parsed = parse_resume(image_path=image_path, model_name=model_name)
    output_path.write_text(
        json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Parsed JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
