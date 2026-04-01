"""Dynamic HTML resume page generator.

This script writes a reusable HTML page that fetches resume data from a JSON
file at runtime (default: `final_resume.json`).
"""

import argparse
import json
from pathlib import Path

DEFAULT_INPUT_PATH = "final_resume.json"
DEFAULT_OUTPUT_PATH = "result.html"


def ensure_json_exists(path: Path) -> None:
    """Validate that input JSON exists and is valid UTF-8 JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    json.loads(path.read_text(encoding="utf-8"))


def build_html(data_url: str) -> str:
    """Return a dynamic HTML page that fetches JSON and renders resume content."""

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Resume</title>
  <style>
    @page {{ size: A4; margin: 14mm; }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      background: #f4f4f4;
      color: #111;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.35;
    }}
    body {{ padding: 24px; }}
    .page {{
      width: 210mm;
      min-height: 297mm;
      margin: 0 auto;
      background: #fff;
      padding: 16mm;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    .toolbar {{
      width: 210mm;
      margin: 0 auto 10px auto;
      display: flex;
      justify-content: flex-end;
    }}
    .save-btn {{
      border: 1px solid #111;
      background: #111;
      color: #fff;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1fr 68mm;
      gap: 16px;
      align-items: start;
    }}
    .main-col {{ min-width: 0; }}
    .side-col {{
      border-left: 1px solid #ddd;
      padding-left: 12px;
    }}
    .hidden {{ display: none !important; }}
    .header {{
      border-bottom: 2px solid #111;
      padding-bottom: 10px;
      margin-bottom: 14px;
    }}
    .name {{ font-size: 28px; font-weight: 700; margin: 0 0 4px 0; }}
    .title {{ font-size: 14px; font-weight: 600; margin: 0 0 8px 0; }}
    .contact {{
      font-size: 11px;
      color: #333;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
    }}
    .contact span {{ white-space: nowrap; }}
    .section {{ margin-top: 14px; page-break-inside: avoid; }}
    .section-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid #bbb;
      padding-bottom: 4px;
      margin-bottom: 8px;
    }}
    .entry {{ margin-bottom: 12px; page-break-inside: avoid; }}
    .removable-entry {{
      position: relative;
      padding-right: 22px;
    }}
    .remove-btn {{
      position: absolute;
      right: 0;
      top: 0;
      width: 18px;
      height: 18px;
      border: 1px solid #aaa;
      background: #fff;
      color: #555;
      border-radius: 3px;
      font-size: 14px;
      line-height: 14px;
      cursor: pointer;
      padding: 0;
    }}
    .entry-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 2px;
    }}
    .entry-role {{ font-size: 12px; font-weight: 700; margin: 0; }}
    .entry-org {{ font-size: 11px; color: #333; margin: 2px 0 0 0; }}
    .entry-date {{ font-size: 11px; color: #333; white-space: nowrap; flex-shrink: 0; }}
    .bullets {{ margin: 6px 0 0 18px; padding: 0; font-size: 11px; }}
    .bullets li {{ margin-bottom: 4px; }}
    .status {{ width: 210mm; margin: 0 auto 10px auto; font-size: 12px; color: #333; }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .page {{ width: auto; min-height: auto; margin: 0; box-shadow: none; padding: 0; }}
      .toolbar, .remove-btn, .status {{ display: none !important; }}
      .side-col {{ display: none !important; }}
      .layout {{ display: block; }}
    }}
  </style>
</head>
<body>
  <div class=\"status\" id=\"status\">Loading resume data...</div>
  <div class=\"toolbar\">
    <button class=\"save-btn\" onclick=\"saveResume()\">Save</button>
  </div>
  <div class=\"page\">
    <header class=\"header\">
      <h1 class=\"name\" id=\"name\">Your Name</h1>
      <p class=\"title\" id=\"job-title\">Target Role</p>
      <div class=\"contact\" id=\"contact\"></div>
    </header>

    <div class=\"layout\">
      <main class=\"main-col\">
        <section class=\"section\">
          <div class=\"section-title\">Experience</div>
          <div id=\"experience-section\"></div>
        </section>

        <section class=\"section\">
          <div class=\"section-title\">Projects</div>
          <div id=\"projects-section\"></div>
        </section>

        <section class=\"section\">
          <div class=\"section-title\">Education</div>
          <div id=\"education-section\"></div>
        </section>
      </main>

      <aside id=\"must-have-panel\" class=\"side-col\">
        <section class=\"section\">
          <div class=\"section-title\">Must-have Skills (JD)</div>
          <div id=\"skills-section\"></div>
        </section>
      </aside>
    </div>
  </div>

  <script>
    const DATA_URL = {json.dumps(data_url)};

    function esc(value) {{
      if (value === null || value === undefined) return '';
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function renderBullets(items) {{
      if (!Array.isArray(items) || !items.length) return '';
      const lines = items.filter(Boolean).map((x) => `<li>${{esc(x)}}</li>`);
      if (!lines.length) return '';
      return `<ul class=\"bullets\">${{lines.join('')}}</ul>`;
    }}

    function renderExperience(entries) {{
      if (!Array.isArray(entries)) return '';
      return entries.map((row) => {{
        const item = (row && row.item) || {{}};
        const role = item.title || '';
        const org = item.company || '';
        const location = item.location || '';
        const start = item.start_date || '';
        const end = item.end_date || 'Present';
        const dateText = [start, end].filter(Boolean).join(' - ');
        const orgLine = location ? `${{org}} | ${{location}}` : org;
        const details = renderBullets(item.details || []);
        return `
          <div class=\"entry removable-entry\">
            <button class=\"remove-btn\" onclick=\"removeEntry(this)\" aria-label=\"Remove entry\">x</button>
            <div class=\"entry-header\">
              <div class=\"entry-left\">
                <p class=\"entry-role\">${{esc(role)}}</p>
                <p class=\"entry-org\">${{esc(orgLine)}}</p>
              </div>
              <div class=\"entry-date\">${{esc(dateText)}}</div>
            </div>
            ${{details}}
          </div>
        `;
      }}).join('');
    }}

    function renderProjects(entries) {{
      if (!Array.isArray(entries)) return '';
      return entries.map((row) => {{
        const item = (row && row.item) || {{}};
        const title = item.title || '';
        const start = item.start_date || '';
        const end = item.end_date || 'Present';
        const dateText = [start, end].filter(Boolean).join(' - ');
        const details = renderBullets(item.details || []);
        return `
          <div class=\"entry removable-entry\">
            <button class=\"remove-btn\" onclick=\"removeEntry(this)\" aria-label=\"Remove entry\">x</button>
            <div class=\"entry-header\">
              <div class=\"entry-left\">
                <p class=\"entry-role\">${{esc(title)}}</p>
                <p class=\"entry-org\">Project</p>
              </div>
              <div class=\"entry-date\">${{esc(dateText)}}</div>
            </div>
            ${{details}}
          </div>
        `;
      }}).join('');
    }}

    function renderEducation(entries) {{
      if (!Array.isArray(entries)) return '';
      return entries.map((item) => {{
        const school = item.school || '';
        const major = item.major || '';
        const start = item.start_date || '';
        const end = item.end_date || 'Present';
        const dateText = [start, end].filter(Boolean).join(' - ');
        return `
          <div class=\"entry\">
            <div class=\"entry-header\">
              <div class=\"entry-left\">
                <p class=\"entry-role\">${{esc(major)}}</p>
                <p class=\"entry-org\">${{esc(school)}}</p>
              </div>
              <div class=\"entry-date\">${{esc(dateText)}}</div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    function setStatus(text) {{
      const el = document.getElementById('status');
      if (el) el.textContent = text;
    }}

    async function loadResume() {{
      try {{
        const res = await fetch(DATA_URL, {{ cache: 'no-store' }});
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        const data = await res.json();

        const target = data.target_job || {{}};
        const stable = data.stable_profile || {{}};
        const basic = stable.basic_info || {{}};
        const personal = stable.personal_info || {{}};

        document.title = `${{basic.name || 'Resume'}} - Resume`;
        document.getElementById('name').textContent = basic.name || 'Your Name';
        document.getElementById('job-title').textContent = target.job_title || 'Target Role';

        const contactParts = [
          basic.location,
          basic.email,
          basic.phone,
          personal.linkedin,
          personal.github,
        ].filter(Boolean);
        document.getElementById('contact').innerHTML = contactParts.map((x) => `<span>${{esc(x)}}</span>`).join('');

        document.getElementById('experience-section').innerHTML = renderExperience(data.selected_experience || []);
        document.getElementById('projects-section').innerHTML = renderProjects(data.selected_projects || []);
        document.getElementById('education-section').innerHTML = renderEducation((stable.education || []));
        document.getElementById('skills-section').innerHTML = renderBullets(target.must_have_skills || []);

        setStatus(`Loaded data from ${{DATA_URL}}`);
      }} catch (err) {{
        setStatus(`Failed to load ${{DATA_URL}}. Run via a local server (not file://). Error: ${{err.message}}`);
      }}
    }}

    function removeEntry(btn) {{
      const entry = btn.closest('.removable-entry');
      if (entry) entry.classList.add('hidden');
    }}

    function saveResume() {{
      const panel = document.getElementById('must-have-panel');
      if (panel) panel.classList.add('hidden');
      const buttons = document.querySelectorAll('.remove-btn');
      buttons.forEach((b) => b.classList.add('hidden'));
      window.print();
    }}

    loadResume();
  </script>
</body>
</html>
"""


def main() -> None:
    """CLI entrypoint for writing dynamic result HTML."""

    parser = argparse.ArgumentParser(description="Render dynamic HTML resume from final_resume.json")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to final resume JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output HTML")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    ensure_json_exists(input_path)

    # Use relative URL so the HTML can fetch JSON from the same project folder.
    data_url = input_path.name
    html_text = build_html(data_url)

    output_path.write_text(html_text, encoding="utf-8")
    print(f"Done. Dynamic HTML rendered to: {output_path.resolve()}")
    print(f"Data source path in HTML: {data_url}")


if __name__ == "__main__":
    main()
