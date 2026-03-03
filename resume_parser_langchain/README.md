# Resume Parser (LangChain + OpenAI Vision)

这个示例用 Python + LangChain 调用 OpenAI 视觉模型，把简历图片解析成结构化 JSON。

## 1) 安装依赖

```bash
cd /Users/gz/Desktop/coding/Resume-Solo/resume_parser_langchain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 `OPENAI_API_KEY`。

## 3) 运行

```bash
python main.py --input /absolute/path/to/resume_image.png --output resume_data.json
```

支持常见图片格式（`png/jpg/jpeg/webp`，由 MIME 自动识别）。

## 4) 输出示例

会输出一个 JSON，包含：

- `name`, `email`, `phone`, `location`
- `summary`, `skills`
- `education[]`
- `experience[]`
- `languages`, `certifications`

## 模型建议

- 质量优先：`MODEL_NAME=gpt-4.1`
- 成本优先：`MODEL_NAME=gpt-4.1-mini`

可在 `.env` 中切换。
