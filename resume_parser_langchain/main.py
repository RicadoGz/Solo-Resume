"""Resume parser entrypoint.

This module reads one or more resume PDFs, asks a structured LLM parser to
extract normalized resume fields, and writes two outputs:
1) `resume_data.json`: the batch result for the current run.
2) `resume_library.json`: an append-only library used by later matching steps.
"""

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


class BasicInfo(BaseModel):
    """Core personal identifiers shown near the top of a resume."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class ExperienceItem(BaseModel):
    """One professional experience record."""

    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class PersonalInfo(BaseModel):
    """Optional profile links and summary text."""

    summary: Optional[str] = None
    website: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class ProjectItem(BaseModel):
    """One project record, usually from personal or school work."""

    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    """One education record."""

    school: Optional[str] = None
    major: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SkillCategory(BaseModel):
    """A named skill group plus its concrete skill terms."""

    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """Top-level structured schema expected from the LLM."""

    basic_info: BasicInfo = Field(default_factory=BasicInfo)
    experience: List[ExperienceItem] = Field(default_factory=list)
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    projects: List[ProjectItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[SkillCategory] = Field(default_factory=list)


def _new_library() -> dict:
    """Return the canonical empty resume-library shape used by this project."""

    return {
        "experience": [],
        "projects": [],
        "education": [],
        "skills": [],
    }


def append_to_library(parsed: ResumeData, library_path: Path) -> dict:
    """Append parsed content into `resume_library.json`.

    Notes:
    - The library is append-only in this MVP and does not deduplicate entries.
    - Missing keys are backfilled to preserve compatibility with older files.
    """

    if not library_path.exists():
        library = _new_library()
    else:
        raw = library_path.read_text(encoding="utf-8").strip()
        library = _new_library() if not raw else json.loads(raw)
        for key, default in _new_library().items():
            library.setdefault(key, default)

    payload = parsed.model_dump()
    library["experience"].extend(payload.get("experience", []))
    library["projects"].extend(payload.get("projects", []))
    library["education"].extend(payload.get("education", []))
    library["skills"].extend(payload.get("skills", []))

    library_path.write_text(
        json.dumps(library, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return library


def to_pdf_data_url(pdf_path: Path) -> str:
    """Encode a local PDF into a `data:` URL for multimodal model input."""

    # Keep MIME detection tolerant and fall back to PDF when unknown.
    mime_type, _ = mimetypes.guess_type(pdf_path.name)
    if mime_type is None:
        mime_type = "application/pdf"

    # The model API expects base64 content when a local file is passed inline.
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def parse_resume(pdf_path: Path, model_name: str) -> ResumeData:
    """Run the structured extraction prompt and return validated `ResumeData`."""

    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(ResumeData)
    data_url = to_pdf_data_url(pdf_path)

    # Two-message pattern:
    # 1) system rules for extraction behavior
    # 2) user task text + PDF file payload
    messages = [
        SystemMessage(
            content=(
                "You are a precise resume parser. "
                "Extract only information visible in the PDF. "
                "Return output strictly in the provided schema. "
                "If missing, use null or empty lists."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Parse this resume into 6 sections: "
                        "basic_info, experience, personal_info, projects, education, skills. "
                        "Education must include school, major, start_date, end_date. "
                        "Experience must include title, company, location, start_date, end_date, details. "
                        "Projects must include title, start_date, end_date, details. "
                        "Skills must include title and skills list. "
                        "Keep original language from resume text in the PDF."
                    ),
                },
                {
                    "type": "file",
                    "file": {"filename": pdf_path.name, "file_data": data_url},
                },
            ]
        ),
    ]

    return structured_llm.invoke(messages)


def main() -> None:
    """CLI entrypoint used by local batch parsing workflow."""

    # Load .env values such as `OPENAI_API_KEY` and optional model override.
    load_dotenv()

    parser = argparse.ArgumentParser(description="Parse resume PDF to structured JSON.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more resume PDF paths.",
    )
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME", "gpt-4.1")
    pdf_paths = [Path(p) for p in args.input]
    output_path = Path("resume_data.json")
    library_path = Path("resume_library.json")

    # Validate all files before making any API calls.
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input file not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input must be a PDF file: {pdf_path}")

    batch_results = []
    for pdf_path in pdf_paths:
        parsed = parse_resume(pdf_path=pdf_path, model_name=model_name)
        batch_results.append({"source_file": str(pdf_path), "data": parsed.model_dump()})
        library = append_to_library(parsed=parsed, library_path=library_path)

    output_path.write_text(
        json.dumps(batch_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Parsed {len(batch_results)} PDF(s). Output: {output_path.resolve()}")
    print(
        "Library updated: "
        f"{library_path.resolve()} "
        f"(exp={len(library['experience'])}, proj={len(library['projects'])}, "
        f"edu={len(library['education'])}, skills={len(library['skills'])})"
    )


if __name__ == "__main__":
    main()
