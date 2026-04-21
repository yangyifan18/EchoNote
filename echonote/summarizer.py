"""LLM summarization — provider abstraction, templates, and long-transcript handling."""

from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
import openai


ACADEMIC_PROMPT = """你是一个专业的学术会议记录整理助手。请根据以下会议转写文本，生成结构化的会议笔记。

## 要求

### 元数据
请从转写内容中提取以下信息（如无法确定，标记为"未知"）：
- 演讲者姓名和机构
- 报告主题
- 相关标签（3-5个，用于知识库分类）

### 摘要
用2-3句话概括本次报告的核心内容和主要贡献。

### 详细内容
这是最重要的部分。请遵循以下原则：
1. **跟随演讲者的原始组织方式** — 如果演讲者按项目/论文时间线讲解，就按项目分块；如果按专题讲解，就按专题分块
2. **不要过度压缩** — 保留具体的数字、方法名、工具名、论文名、实验结果等细节
3. **标注论文外信息** — 用「📌 论文外」标记演讲者分享的、在已发表论文中未涉及的实践经验
4. **保持足够上下文** — 使后续读者（或AI助手）能基于这份笔记进行深入问答

每个分块用 ### 标题标记，内容用 bullet points 组织。

### Q&A
如果转写中包含提问环节，按 Q1/Q2/... 格式整理每个问答对。如果没有明显的提问环节，省略此部分。

### 行动项
提取可执行的TODO事项（如需要调研的工具、需要联系的人、需要阅读的论文等），用 `- [ ]` 格式。如果没有明确的行动项，省略此部分。

## 输出格式

请直接输出 Markdown 格式。以 YAML frontmatter 开头：

---
speaker: 姓名 (机构)
topic: 主题
tags: [tag1, tag2, ...]
---

## 摘要
...

## 详细内容
### 分块标题1
...

## Q&A
### Q1: 问题
...

## 行动项
- [ ] ...

---

## 转写文本

{transcript}"""


MEETING_PROMPT = """你是一个专业的会议纪要整理助手。请根据以下会议转写文本，生成结构化的会议笔记。

## 要求

### 元数据
请从转写内容中提取以下信息（如无法确定，标记为"未知"）：
- 与会人
- 会议主题
- 相关标签（3-5个，用于知识库分类）

### 摘要
用2-3句话概括这次会议的主要目标、结论和进展。

### 详细内容
按会议的原始推进顺序整理，通常优先按议题、决策流或项目模块分块。
每个分块用 ### 标题标记，内容用 bullet points 组织，并保留关键背景、数字、时间点、依赖关系和风险。

### 决策
列出已经明确做出的决策。如果没有明确决策，省略此部分。

### 行动项
提取可执行的TODO事项，用 `- [ ]` 格式；如果能从上下文推断负责人或截止时间，一并写出。

### 风险与阻塞
整理尚未解决的风险、争议和阻塞。如果没有，省略此部分。

## 输出格式

请直接输出 Markdown 格式。以 YAML frontmatter 开头：

---
participants: [姓名1, 姓名2]
topic: 主题
tags: [meeting, ...]
---

## 摘要
...

## 详细内容
### 议题 1
...

## 决策
- ...

## 行动项
- [ ] ...

## 风险与阻塞
- ...

---

## 转写文本

{transcript}"""


PROMPT_TEMPLATES = {
    "academic": ACADEMIC_PROMPT,
    "meeting": MEETING_PROMPT,
}


def load_prompt_template(template_name: str = "academic", prompt_path: str = "") -> str:
    """Load a built-in or file-based prompt template."""
    if prompt_path:
        template = Path(prompt_path).expanduser().read_text(encoding="utf-8")
    else:
        try:
            template = PROMPT_TEMPLATES[template_name]
        except KeyError as exc:
            supported = ", ".join(sorted(PROMPT_TEMPLATES))
            raise ValueError(f"Unknown prompt template: {template_name}. Supported templates: {supported}") from exc

    if "{transcript}" not in template:
        raise ValueError("Prompt template must contain a {transcript} placeholder.")

    return template


def build_prompt(
    transcript: str,
    template_name: str = "academic",
    prompt_path: str = "",
) -> str:
    """Construct the summarization prompt with the transcript inserted."""
    template = load_prompt_template(template_name=template_name, prompt_path=prompt_path)
    return template.replace("{transcript}", transcript)


def split_transcript(transcript: str, max_chars_per_chunk: int) -> list[str]:
    """Split a long transcript into roughly bounded chunks while preserving order."""
    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be positive.")

    cleaned = transcript.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    current = ""
    for line in cleaned.splitlines(keepends=True):
        remaining = line
        while remaining:
            space_left = max_chars_per_chunk - len(current)
            if space_left <= 0:
                chunks.append(current.strip())
                current = ""
                space_left = max_chars_per_chunk

            if len(remaining) <= space_left:
                current += remaining
                remaining = ""
            else:
                current += remaining[:space_left]
                remaining = remaining[space_left:]
                chunks.append(current.strip())
                current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _template_guidance(template: str) -> str:
    """Render a template as guidance for chunk-level and merge prompts."""
    return template.replace("{transcript}", "[transcript goes here]")


def build_chunk_prompt(chunk_transcript: str, final_template: str, chunk_index: int, total_chunks: int) -> str:
    """Prompt for summarizing one chunk of a longer transcript."""
    return f"""你正在整理一份超长会议转写的中间片段笔记。

最终输出必须遵守下面的模板和风格要求，但你现在只能根据当前片段写中间摘要。
- 只总结当前片段里明确出现的信息
- 保留专有名词、数字、时间点、决策、问答和行动项线索
- 不要补全跨片段结论，也不要假设未出现的信息

## 最终模板与风格要求
{_template_guidance(final_template)}

## 当前片段
这是第 {chunk_index}/{total_chunks} 个片段。

{chunk_transcript}"""


def build_synthesis_prompt(chunk_summaries: list[str], final_template: str) -> str:
    """Prompt for merging chunk summaries into one final note."""
    rendered_notes = "\n\n".join(
        f"### Chunk {index}\n{summary}" for index, summary in enumerate(chunk_summaries, start=1)
    )
    return f"""你正在把同一场会议的分段笔记合并成最终笔记。

请严格遵守下面的模板和风格要求，对分段笔记去重、合并并纠正局部冲突。
- 只基于分段笔记输出
- 保留具体细节，不要过度压缩
- 如果分段笔记之间信息不完整或冲突，保守表达，不要编造

## 最终模板与风格要求
{_template_guidance(final_template)}

## 分段笔记
{rendered_notes}"""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return the response text."""


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def _make_provider(provider_name: str, model: str, api_key: str) -> LLMProvider:
    if provider_name == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM provider: {provider_name}")


def summarize_with_provider(
    transcript: str,
    provider: LLMProvider,
    template_name: str = "academic",
    prompt_path: str = "",
    max_chars_per_chunk: int = 12000,
) -> str:
    """Summarize a transcript with a ready-made provider instance."""
    final_template = load_prompt_template(template_name=template_name, prompt_path=prompt_path)
    chunks = split_transcript(transcript, max_chars_per_chunk)
    if not chunks:
        raise ValueError("Transcript is empty.")

    if len(chunks) == 1:
        return provider.generate(build_prompt(chunks[0], template_name=template_name, prompt_path=prompt_path))

    chunk_summaries = []
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = build_chunk_prompt(chunk, final_template, index, total_chunks)
        chunk_summaries.append(provider.generate(chunk_prompt))

    return provider.generate(build_synthesis_prompt(chunk_summaries, final_template))


def summarize(
    transcript: str,
    provider_name: str,
    model: str,
    api_key: str,
    template_name: str = "academic",
    prompt_path: str = "",
    max_chars_per_chunk: int = 12000,
) -> str:
    """Summarize a transcript using the configured LLM provider."""
    provider = _make_provider(provider_name, model, api_key)
    return summarize_with_provider(
        transcript=transcript,
        provider=provider,
        template_name=template_name,
        prompt_path=prompt_path,
        max_chars_per_chunk=max_chars_per_chunk,
    )
