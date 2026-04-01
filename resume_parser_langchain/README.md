# Resume Tailoring Pipeline (LangChain + OpenAI)
# 简历定制流水线（LangChain + OpenAI）

This project parses resume PDFs, analyzes a job description (JD), runs RAG-style matching against your resume library, and generates a tailored `final_resume.json` plus a dynamic `result.html`.

本项目用于解析 PDF 简历、分析岗位 JD、基于简历库做 RAG 匹配，最终生成定制化 `final_resume.json` 和动态 `result.html`。

## 0) Detailed Project Description (Resume-Ready) | 最详细项目描述（可直接放简历）

### 中文版（详细）

项目名称：AI 驱动的简历定制与岗位匹配系统（Resume Tailoring Pipeline）  
技术栈：Python、LangChain、OpenAI API、Pydantic、SQLite、HTML/CSS/JavaScript

这是一个面向真实求职场景的端到端简历智能改写与匹配项目，目标是把“岗位描述（JD）→简历筛选→定制输出”流程自动化，减少手动改简历成本并提升岗位相关性。系统从 PDF 简历和 JD 文本出发，通过结构化抽取、RAG 检索、排序去重、最终组装与前端动态渲染，生成可直接投递的目标岗位版本简历。

核心技术实现与工程设计：

- 结构化简历解析：使用 LangChain + OpenAI 对 PDF 简历执行结构化抽取，输出 6 大模块（`basic_info / experience / personal_info / projects / education / skills`），并通过 Pydantic 做字段级类型校验与默认值兜底，保证输出稳定可复用。
- 简历知识库构建：将多份简历解析结果追加写入 `resume_library.json`，按经历/项目/教育/技能分区组织，形成可持续扩展的候选素材池，支持后续多 JD 复用匹配。
- JD 结构化分析：将 `jd.txt` 抽取为 `job_title`、`must_have_skills`、`keywords`、`soft_skills`、`required_experience` 等可计算信号，为检索与排序提供统一输入。
- 双通道 RAG 匹配：
  - Keyword 检索：基于标准化文本、短语匹配与 token fallback，结合命中数（`keyword_score`）和覆盖率（`coverage`）做排序。
  - Embedding 检索：基于 OpenAI Embeddings + cosine similarity 做语义召回，支持阈值控制与 Top-K 选择，补足关键词匹配的语义缺失。
- 去重与结果质量控制：在 keyword 模式用 Jaccard（token set）去重，在 embedding 模式用向量相似度去重，避免重复经历占位，提高最终简历可读性。
- 向量缓存优化：使用 SQLite 本地缓存向量，结合 `library_hash + embedding_model` 判定是否重建，避免重复 embedding 计算，降低重跑时延与 API 成本。
- 最终简历组装：把稳定信息（基础信息/教育）与高相关匹配项（经历/项目）合并输出为 `final_resume.json`，为渲染层提供干净、可追踪的数据边界。
- 动态前端渲染：`result.html` 通过 `fetch(final_resume.json)` 动态渲染内容，支持手动移除条目、打印导出，形成“匹配→预览→投递”闭环。

项目成果与价值（可用于简历表达）：

- 将原本手动 JD 对照改简历流程产品化为自动流水线，显著提升定制效率和迭代速度。
- 支持多检索策略（keyword/embedding）和可调参数（阈值、Top-K、去重相似度），便于针对不同岗位类型做精细化优化。
- 提供结构化、可解释的匹配输出（命中关键词、覆盖率、相似度），便于人工审核与快速修改。
- 形成“数据层（JSON）-检索层（RAG）-展示层（HTML）”分层架构，具备继续扩展为 Web 服务/API 的工程基础。

可直接放简历的高密度版本（中文）：

- 设计并实现 AI 简历定制流水线（PDF 解析→JD 抽取→RAG 匹配→动态渲染），基于 Python/LangChain/OpenAI/Pydantic/SQLite 构建，支持 keyword+embedding 双检索、Top-K 与相似度去重，输出可直接投递的岗位定制简历。  
- 构建本地向量缓存与哈希增量更新机制，避免重复 embedding，降低多次重跑时延与调用成本；同时提供可解释匹配指标（命中词/覆盖率/相似度）以支持人工快速决策。  

### English Version (Detailed)

Project: AI-Powered Resume Tailoring and JD Matching Pipeline  
Tech Stack: Python, LangChain, OpenAI API, Pydantic, SQLite, HTML/CSS/JavaScript

This project automates the end-to-end workflow of role-targeted resume tailoring. Starting from PDF resumes and a job description, it performs structured extraction, RAG-style retrieval, ranking/deduplication, final JSON composition, and dynamic HTML rendering to produce a job-aligned resume version ready for review and submission.

Core implementation highlights:

- Structured resume parsing: Implemented schema-constrained extraction with LangChain + OpenAI and validated output via Pydantic, covering six normalized sections (`basic_info`, `experience`, `personal_info`, `projects`, `education`, `skills`).
- Reusable resume library: Built an append-based candidate pool in `resume_library.json` so multiple resumes can be accumulated and reused across different JD matching runs.
- JD signal extraction: Converted raw `jd.txt` into machine-usable hiring signals (`must_have_skills`, `keywords`, `soft_skills`, `required_experience`, etc.) for deterministic downstream ranking.
- Dual retrieval strategy:
  - Keyword retrieval using normalized phrase/token matching, ranked by hit count and query coverage.
  - Embedding retrieval using OpenAI vectors + cosine similarity thresholding for semantic matching.
- Quality control and dedup: Applied Jaccard-based near-duplicate filtering in keyword mode and vector-similarity dedup in embedding mode to improve result diversity and readability.
- Embedding cache optimization: Added SQLite vector caching with hash/model-based invalidation (`library_hash`, `embedding_model`) to avoid unnecessary re-embedding and reduce rerun cost/latency.
- Final payload composition: Merged stable profile fields with selected high-relevance experience/project items into `final_resume.json`, preserving clean data boundaries between extraction, retrieval, and presentation.
- Dynamic rendering UX: Delivered a runtime HTML renderer (`fetch(final_resume.json)`) with quick manual pruning and print export, enabling a fast “match -> review -> submit” workflow.

Impact summary (resume-friendly):

- Productized a previously manual JD-to-resume tailoring process into a repeatable AI pipeline.
- Improved iteration speed for role-specific resume generation while keeping transparent, explainable matching signals.
- Established a layered architecture (data, retrieval, presentation) that can be extended into service/API deployment.

Resume-ready compact version (English):

- Built an end-to-end AI resume-tailoring pipeline (PDF parsing -> JD extraction -> RAG matching -> dynamic rendering) using Python, LangChain, OpenAI, Pydantic, and SQLite; implemented dual retrieval (keyword + embedding), Top-K ranking, and similarity-based deduplication to generate role-targeted resumes.  
- Added hash-driven embedding cache invalidation and structured match diagnostics (hit keywords, coverage, similarity), reducing rerun cost and enabling fast human-in-the-loop review.  

## 1) Project Structure | 项目结构

Key files:

- `main.py`: Parse one or more resume PDFs into structured data and append to library.
- `read_jd.py`: Analyze `jd.txt` into structured hiring signals.
- `match_rag.py`: Match JD signals to library items (keyword or embedding mode).
- `build_final_json.py`: Merge stable profile + JD analysis + match results into `final_resume.json`.
- `render_resume_html.py`: Generate dynamic HTML page (`result.html`) that fetches `final_resume.json` at runtime.

核心文件说明：

- `main.py`：解析一个或多个 PDF 简历，输出结构化数据并追加到简历库。
- `read_jd.py`：把 `jd.txt` 抽取成结构化岗位要求。
- `match_rag.py`：将 JD 要求与简历库匹配（支持 keyword / embedding）。
- `build_final_json.py`：合并基础信息 + JD 分析 + 匹配结果，生成 `final_resume.json`。
- `render_resume_html.py`：生成动态 `result.html`，页面运行时读取 `final_resume.json`。

## 2) Setup | 环境准备

```bash
cd /Users/gz/Desktop/coding/Resume-Solo/resume_parser_langchain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your API key in `.env`:

```env
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4.1
JD_MODEL_NAME=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

在 `.env` 中至少填写 `OPENAI_API_KEY`。

## 3) End-to-End Run | 全流程运行

### Step A: Parse resume PDF(s) | 解析简历 PDF

```bash
python3 main.py --input /absolute/path/to/resume1.pdf /absolute/path/to/resume2.pdf
```

Outputs:

- `resume_data.json` (batch parse output)
- `resume_library.json` (append-only library)

输出：

- `resume_data.json`（本次批量解析结果）
- `resume_library.json`（可复用的简历库）

### Step B: Analyze JD | 分析岗位 JD

```bash
python3 read_jd.py --jd jd.txt --output jd_analysis.json
```

### Step C: Match JD to library (RAG) | RAG 匹配

Keyword mode (default):

```bash
python3 match_rag.py --top-k 5 --dedup-similarity 0.9 --retrieval keyword
```

Embedding mode:

```bash
python3 match_rag.py --top-k 5 --dedup-similarity 0.9 --retrieval embedding --threshold 0.38
```

Output: `rag_match_output.json`

### Step D: Build final resume JSON | 生成最终简历 JSON

```bash
python3 build_final_json.py \
  --resume-data resume_data.json \
  --library resume_library.json \
  --jd-analysis jd_analysis.json \
  --match rag_match_output.json \
  --output final_resume.json
```

### Step E: Render dynamic HTML | 生成动态 HTML

```bash
python3 render_resume_html.py --input final_resume.json --output result.html
```

`result.html` loads data from `final_resume.json` via `fetch()`.
`result.html` 通过 `fetch()` 动态读取 `final_resume.json`。

## 4) View HTML Correctly | 正确打开 HTML

Do **not** open by double-click (`file://`) because browser may block `fetch()`.
不要直接双击打开（`file://`），否则浏览器可能拦截 `fetch()`。

Use local server:

```bash
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/result.html`

## 5) Minimal Quick Commands | 最短命令流程

If your library already exists and you only want a new final JSON for a new JD:
如果简历库已有，仅针对新 JD 生成新的 final JSON：

```bash
python3 read_jd.py --jd jd.txt --output jd_analysis.json
python3 match_rag.py --top-k 5 --dedup-similarity 0.9 --retrieval keyword
python3 build_final_json.py --output final_resume.json
python3 render_resume_html.py --input final_resume.json --output result.html
```

## 6) Common Issues | 常见问题

1. `result.html` shows empty sections  
   Usually `fetch` failed due `file://` open mode. Use `python3 -m http.server`.

2. No match results in `rag_match_output.json`  
   Check JD quality (`jd.txt`) and try `--retrieval embedding` with a lower threshold (for example `0.32`).

3. API errors  
   Verify `.env` API key and model names.

## 7) Notes | 说明

- `resume_library.json` is append-only in current MVP (no dedup at ingestion).
- Matching stage performs dedup on retrieved candidates using similarity threshold.
- `build_final_json.py` keeps stable profile fields and injects selected experience/projects.

- 当前 MVP 中 `resume_library.json` 是追加模式（导入时不去重）。
- 匹配阶段会对候选结果做相似度去重。
- `build_final_json.py` 会保留稳定信息，并注入筛选后的经历/项目。
