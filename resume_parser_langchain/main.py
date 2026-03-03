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
    # Basic profile identity / 基本身份信息
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class ExperienceItem(BaseModel):
    # Experience: role, time, location, company, and details
    # 工作经历：职位、时间、地点、公司、做了什么
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class PersonalInfo(BaseModel):
    # Personal section / 个人信息部分（补充信息）
    summary: Optional[str] = None
    website: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class ProjectItem(BaseModel):
    # Project: title, time, and what you did
    # 项目经历：标题、时间、项目内容
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    # Education: school, major, and time
    # 教育经历：学校、专业、时间
    school: Optional[str] = None
    major: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SkillCategory(BaseModel):
    # Skills: a title/category and concrete skill list
    # 专业技能：一个标题/分类 + 具体技能列表
    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    # 6 required sections / 你要求的六大部分
    basic_info: BasicInfo = Field(default_factory=BasicInfo)
    experience: List[ExperienceItem] = Field(default_factory=list)
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    projects: List[ProjectItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[SkillCategory] = Field(default_factory=list)


def to_pdf_data_url(pdf_path: Path) -> str:
    """
    Convert a local PDF file to a data URL string.
    把本地 PDF 文件转成 data URL 字符串（给模型直接读取 PDF 内容）。

    Why needed / 为什么需要这一步：
    - We assume resume input is always PDF.
      这里假设简历输入永远是 PDF。
    - Using `data:application/pdf;base64,...` lets us send local files
      without hosting them on a public URL.
      用 data URL 可以直接发送本地文件，不需要先上传到公网链接。
    """
    # Guess MIME type; fallback to PDF.
    # 推断 MIME 类型；无法推断时兜底为 PDF。
    mime_type, _ = mimetypes.guess_type(pdf_path.name)
    if mime_type is None:
        mime_type = "application/pdf"

    # Read binary bytes from local file
    # 读取 PDF 的二进制内容
    pdf_bytes = pdf_path.read_bytes()
    # Convert bytes to base64 text for embedding in data URL
    # 把二进制转为 base64 文本，拼进 data URL
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    # Final format: data:<mime>;base64,<content>
    # 最终格式：data:<mime>;base64,<内容>
    return f"data:{mime_type};base64,{b64}"


def parse_resume(pdf_path: Path, model_name: str) -> ResumeData:
    """
    Run the LangChain parsing pipeline and return typed resume data.
    执行 LangChain 解析流程，并返回类型化的简历数据对象。

    Flow / 流程：
    1) Build model client
    2) Bind output schema (ResumeData)
    3) Build messages (instruction + PDF)
    4) Invoke model and get validated result
    """
    # Create chat model client
    # 创建聊天模型客户端
    llm = ChatOpenAI(model=model_name, temperature=0)
    # Attach schema so model output is parsed/validated as ResumeData
    # 绑定结构化输出，要求模型返回可被 ResumeData 校验的结果
    structured_llm = llm.with_structured_output(ResumeData)
    # Convert local PDF to inline data URL for message payload
    # 把本地 PDF 转成 data URL，放进消息体
    data_url = to_pdf_data_url(pdf_path)

    # We send two message roles:
    # - SystemMessage: global behavior/rules
    # - HumanMessage: task instruction + actual PDF
    # 这里发送两类消息：
    # - SystemMessage：全局规则
    # - HumanMessage：任务说明 + 真实 PDF
    messages = [
        SystemMessage(
            content=(
                "You are a precise resume parser. "
                "Extract only information visible in the image. "
                "Return output strictly in the provided schema. "
                "If missing, use null or empty lists."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        # Explicitly tell model which sections/fields are required
                        # 明确告诉模型必须输出哪些 section/字段
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
                    # Multimodal PDF input
                    # 多模态 PDF 输入
                    "type": "file",
                    "file": {"filename": pdf_path.name, "file_data": data_url},
                },
            ]
        ),
    ]

    # Execute call and return Pydantic object (ResumeData)
    # 执行调用并返回 Pydantic 对象（ResumeData）
    return structured_llm.invoke(messages)


def main() -> None:
    """
    CLI entrypoint.
    命令行入口函数：读取参数 -> 调用解析 -> 写出 JSON。
    """
    # Load variables from .env (e.g., API key, model name)
    # 从 .env 加载环境变量（比如 API key、模型名）
    load_dotenv()

    # Define command-line interface arguments
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Parse resume PDF to structured JSON.")
    parser.add_argument("--input", required=True, help="Path to resume PDF file.")
    parser.add_argument(
        "--output",
        default="resume_data.json",
        help="Path to output JSON file (default: resume_data.json).",
    )
    args = parser.parse_args()

    # Read model config from env; default to gpt-4.1
    # 从环境变量读模型名；默认 gpt-4.1
    model_name = os.getenv("MODEL_NAME", "gpt-4.1")
    # Convert CLI paths to Path objects for safer file operations
    # 把路径参数转成 Path 对象，便于安全处理文件
    pdf_path = Path(args.input)
    output_path = Path(args.output)

    # Basic input validation
    # 输入文件存在性检查
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file: {pdf_path}")

    # Core parsing call
    # 核心解析调用
    parsed = parse_resume(pdf_path=pdf_path, model_name=model_name)
    # Serialize to pretty JSON; keep Chinese/non-ASCII characters
    # 输出美化 JSON；保留中文等非 ASCII 字符
    output_path.write_text(
        json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print output path for quick check
    # 打印输出文件路径，方便你立即查看
    print(f"Done. Parsed JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
