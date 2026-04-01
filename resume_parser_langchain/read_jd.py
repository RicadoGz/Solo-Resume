"""Job-description analyzer.

This module reads a plain-text JD and extracts structured hiring signals for
later matching, including must-have skills, keywords, and responsibilities.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

DEFAULT_JD_PATH = "jd.txt"
DEFAULT_OUTPUT_PATH = "jd_analysis.json"


class JobAnalysis(BaseModel):
    """Structured representation of a job description used downstream."""

    job_title: Optional[str] = None
    company: Optional[str] = None
    seniority: Optional[str] = None
    employment_type: Optional[str] = None
    location_mode: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    required_experience: List[str] = Field(default_factory=list)
    education_requirements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    ats_keywords: List[str] = Field(default_factory=list)
    rejection_signals: List[str] = Field(default_factory=list)


def analyze_jd(jd_path: Path, model_name: str) -> JobAnalysis:
    """Extract `JobAnalysis` from a `.txt` job description using structured output."""

    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(JobAnalysis)

    # The system prompt keeps the model focused on extraction only.
    system_prompt = (
        "You are a job description extractor. "
        "Extract structured hiring requirements from the JD only. "
        "Do not recommend candidate decisions. Do not invent missing info."
    )

    jd_text = jd_path.read_text(encoding="utf-8")

    # The user payload includes a clear extraction checklist plus raw JD text.
    jd_payload = {"type": "text", "text": f"Job Description:\n{jd_text}"}
    user_text = (
        "The JD is provided as plain text (.txt).\n"
        "Extract: keywords, must-have skills, nice-to-have skills, soft skills, "
        "responsibilities, required experience, education/certifications, ATS keywords, "
        "rejection_signals."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                jd_payload,
            ]
        ),
    ]
    return structured_llm.invoke(messages)


def main() -> None:
    """CLI entrypoint for reading one JD and writing `jd_analysis.json`."""

    load_dotenv()

    parser = argparse.ArgumentParser(description="Read and analyze one job description.")
    parser.add_argument(
        "--jd",
        default=DEFAULT_JD_PATH,
        help=f"Path to JD txt file (default: {DEFAULT_JD_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to analysis output JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    jd_path = Path(args.jd)
    output_path = Path(args.output)

    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")
    if jd_path.suffix.lower() != ".txt":
        raise ValueError(f"JD must be a .txt file: {jd_path}")

    # Use a smaller default model for lower cost while preserving reliability.
    model_name = os.getenv("JD_MODEL_NAME", "gpt-4.1-mini")

    analysis = analyze_jd(jd_path=jd_path, model_name=model_name)

    output_path.write_text(
        json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. JD analysis saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
