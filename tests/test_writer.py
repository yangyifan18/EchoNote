from pathlib import Path
from meetfocus.writer import write_summary, write_transcript


def test_write_summary_creates_markdown_file(tmp_path):
    vault_dir = tmp_path / "vault" / "Meetings"
    summary_content = "---\nspeaker: 张教授 (XX研究院)\ntopic: 数据工程\ntags: [meeting, data]\n---\n\n## 摘要\n测试总结"

    result_path = write_summary(
        summary=summary_content,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="数据工程报告",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert result_path.name == "2026-04-10-数据工程报告.md"
    content = result_path.read_text(encoding="utf-8")
    assert "张教授" in content
    assert "测试总结" in content


def test_write_summary_adds_date_frontmatter(tmp_path):
    summary_content = "---\nspeaker: 测试\ntopic: 测试\ntags: [test]\n---\n\n## 摘要\n内容"

    result_path = write_summary(
        summary=summary_content,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="测试",
        filename_format="{date}-{title}",
    )

    content = result_path.read_text(encoding="utf-8")
    assert "date: 2026-04-10" in content


def test_write_summary_creates_folder_if_missing(tmp_path):
    vault_dir = tmp_path / "vault"

    result_path = write_summary(
        summary="---\ntopic: test\n---\n\n## 摘要\ntest",
        vault_path=str(vault_dir),
        folder="Meetings",
        date="2026-04-10",
        title="test",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert (vault_dir / "Meetings").is_dir()


def test_write_transcript(tmp_path):
    vault_dir = tmp_path / "vault" / "Meetings"
    transcript = "这是完整的转写文本。\n第二段内容。"

    result_path = write_transcript(
        transcript=transcript,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="测试",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert result_path.name == "2026-04-10-测试-transcript.md"
    content = result_path.read_text(encoding="utf-8")
    assert "转写文本" in content
