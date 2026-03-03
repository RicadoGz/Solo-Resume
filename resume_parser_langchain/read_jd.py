import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class JobAnalysis(BaseModel):
    # Core JD signals for resume generation / 用于后续生成简历的 JD 核心信号
    job_title: Optional[str] = None
    company: Optional[str] = None
    seniority: Optional[str] = None
    employment_type: Optional[str] = None
    location_mode: Optional[str] = None
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    required_experience: List[str] = Field(default_factory=list)
    education_requirements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    ats_keywords: List[str] = Field(default_factory=list)
    resume_focus_points: List[str] = Field(default_factory=list)
    recommended_experience_titles: List[str] = Field(default_factory=list)
    recommended_project_titles: List[str] = Field(default_factory=list)
    recommended_skill_categories: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)


def _default_library() -> Dict[str, list]:
    return {
        "experience": [],
        "projects": [],
        "education": [],
        "skills": [],
    }


def load_library(library_path: Path) -> Dict[str, list]:
    if not library_path.exists():
        return _default_library()
    raw = library_path.read_text(encoding="utf-8").strip()
    data = _default_library() if not raw else json.loads(raw)
    for key, value in _default_library().items():
        data.setdefault(key, value)
    return data


def build_library_summary(library: Dict[str, list]) -> Dict[str, list]:
    # Keep a compact summary to reduce token usage / 压缩库内容，减少 token 消耗
    experience_titles = []
    for item in library.get("experience", []):
        title = (item or {}).get("title")
        if title:
            experience_titles.append(title)

    project_titles = []
    for item in library.get("projects", []):
        title = (item or {}).get("title")
        if title:
            project_titles.append(title)

    skill_categories = []
    skill_terms = []
    for item in library.get("skills", []):
        title = (item or {}).get("title")
        if title:
            skill_categories.append(title)
        for skill in (item or {}).get("skills", []):
            if skill:
                skill_terms.append(skill)

    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "experience_titles": dedupe(experience_titles)[:120],
        "project_titles": dedupe(project_titles)[:120],
        "skill_categories": dedupe(skill_categories)[:120],
        "skill_terms": dedupe(skill_terms)[:400],
    }


def to_data_url(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(file_path.name)
    if mime_type is None:
        mime_type = "application/pdf"
    file_bytes = file_path.read_bytes()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def analyze_jd(jd_path: Path, library_summary: Dict[str, list], model_name: str) -> JobAnalysis:
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(JobAnalysis)

    system_prompt = (
        "You are a job description analyst for resume generation. "
        "Extract practical requirements from the JD. "
        "Also map JD needs to candidate library summary and propose best matches. "
        "Be strict and avoid inventing facts."
    )

    if jd_path.suffix.lower() == ".pdf":
        jd_payload = {
            "type": "file",
            "file": {"filename": jd_path.name, "file_data": to_data_url(jd_path)},
        }
        jd_hint = "The JD is provided as a PDF file."
    else:
        jd_text = jd_path.read_text(encoding="utf-8")
        jd_payload = {"type": "text", "text": f"Job Description:\n{jd_text}"}
        jd_hint = "The JD is provided as plain text."

    user_text = (
        f"{jd_hint}\n"
        "Extract: role, must-have skills, nice-to-have skills, responsibilities, "
        "required experience, education/certifications, ATS keywords.\n"
        "Then use library summary to recommend which experience/project/skill categories "
        "should be emphasized in the next generated resume.\n"
        f"Library summary JSON:\n{json.dumps(library_summary, ensure_ascii=False)}"
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
    load_dotenv()

    parser = argparse.ArgumentParser(description="Read and analyze one job description.")
    parser.add_argument("--jd", required=True, help="Path to JD file (.txt/.md/.pdf).")
    parser.add_argument(
        "--library",
        default="resume_library.json",
        help="Path to resume library JSON (default: resume_library.json).",
    )
    parser.add_argument(
        "--output",
        default="jd_analysis.json",
        help="Path to analysis output JSON (default: jd_analysis.json).",
    )
    args = parser.parse_args()

    jd_path = Path(args.jd)
    library_path = Path(args.library)
    output_path = Path(args.output)

    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    # OpenAI API has no truly free model; use a low-cost model by default.
    # OpenAI API 没有真正免费的模型；这里默认使用低成本模型。
    model_name = os.getenv("JD_MODEL_NAME", "gpt-4.1-mini")

    library = load_library(library_path)
    library_summary = build_library_summary(library)
    analysis = analyze_jd(jd_path=jd_path, library_summary=library_summary, model_name=model_name)

    output_path.write_text(
        json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. JD analysis saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
