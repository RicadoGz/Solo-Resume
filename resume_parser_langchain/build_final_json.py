"""Final resume payload builder.

This module merges:
- stable profile data from `resume_data.json`
- JD analysis signals
- matched experience/project items from RAG output

The result is written to `final_resume.json` and consumed by the HTML renderer.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_RESUME_DATA_PATH = "resume_data.json"
DEFAULT_LIBRARY_PATH = "resume_library.json"
DEFAULT_JD_ANALYSIS_PATH = "jd_analysis.json"
DEFAULT_MATCH_PATH = "rag_match_output.json"
DEFAULT_OUTPUT_PATH = "final_resume.json"


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON and fail fast if the file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pick_base_profile(resume_data: Any) -> Dict[str, Any]:
    """Return the base profile object from supported `resume_data.json` shapes.

    Supported formats:
    - direct object: `{...}`
    - batch list: `[{"source_file": ..., "data": {...}}, ...]`
      In batch mode we intentionally use the latest entry.
    """

    if isinstance(resume_data, dict):
        return resume_data
    if isinstance(resume_data, list) and resume_data:
        last = resume_data[-1]
        if isinstance(last, dict) and isinstance(last.get("data"), dict):
            return last["data"]
    return {}


def parse_doc_id(doc_id: str) -> Tuple[str, int]:
    """Parse IDs like `experience:12` into `(prefix, index)`.

    Returns `("", -1)` when parsing fails, so callers can skip invalid rows.
    """

    if ":" not in doc_id:
        return "", -1
    prefix, index_str = doc_id.split(":", 1)
    try:
        return prefix, int(index_str)
    except ValueError:
        return prefix, -1


def build_selected_items(match_items: List[dict], library: Dict[str, Any], source_key: str) -> List[dict]:
    """Build final selected rows by joining match metadata with library content."""

    selected = []
    source_list = library.get(source_key, [])
    for m in match_items:
        doc_id = m.get("doc_id", "")
        prefix, idx = parse_doc_id(doc_id)

        # Example: source_key="projects" should accept prefix="project".
        if prefix != source_key.rstrip("s") or idx < 0 or idx >= len(source_list):
            continue

        selected.append(
            {
                "match_meta": {
                    "doc_id": m.get("doc_id"),
                    "keyword_score": m.get("keyword_score"),
                    "coverage": m.get("coverage"),
                    "matched_keywords": m.get("matched_keywords", []),
                },
                "item": source_list[idx],
            }
        )
    return selected


def main() -> None:
    """CLI entrypoint for composing the final JSON used by HTML rendering."""

    parser = argparse.ArgumentParser(description="Build final resume JSON from static + matched data.")
    parser.add_argument("--resume-data", default=DEFAULT_RESUME_DATA_PATH)
    parser.add_argument("--library", default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--jd-analysis", default=DEFAULT_JD_ANALYSIS_PATH)
    parser.add_argument("--match", default=DEFAULT_MATCH_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    resume_data = load_json(Path(args.resume_data))
    library = load_json(Path(args.library))
    jd = load_json(Path(args.jd_analysis))
    match_output = load_json(Path(args.match))

    base = pick_base_profile(resume_data)

    experience_matches = match_output.get("experience_results", [])
    project_matches = match_output.get("project_results", [])

    # Backward compatibility for older output that used one generic `results` key.
    if not experience_matches and not project_matches and isinstance(match_output.get("results"), list):
        experience_matches = match_output.get("results", [])

    final_payload = {
        "target_job": {
            "job_title": jd.get("job_title"),
            "keywords": jd.get("keywords", []),
            "must_have_skills": jd.get("must_have_skills", []),
            "soft_skills": jd.get("soft_skills", []),
        },
        "stable_profile": {
            "basic_info": base.get("basic_info", {}),
            "personal_info": base.get("personal_info", {}),
            "education": base.get("education", []),
        },
        "selected_experience": build_selected_items(experience_matches, library, "experience"),
        "selected_projects": build_selected_items(project_matches, library, "projects"),
    }

    Path(args.output).write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Final JSON saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
