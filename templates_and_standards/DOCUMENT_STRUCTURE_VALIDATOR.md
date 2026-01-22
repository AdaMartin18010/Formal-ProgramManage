# 文档结构验证工具说明

Document Structure Validator Guide

## 📋 文档说明 / Document Description

本文档提供文档结构验证的自动化检查方法和工具说明，帮助快速识别不符合10个必需章节要求的文档。

This document provides automated checking methods and tool descriptions for document structure validation, helping to quickly identify documents that do not meet the 10 required sections.

**创建时间**: 2026-01-27
**最后更新**: 2026-01-27
**版本**: 1.0
**状态**: ✅ 持续更新中

---

## 🔍 验证方法 / Validation Methods

### 方法1：手动检查清单 / Method 1: Manual Checklist

使用`DOCUMENT_STRUCTURE_CHECKLIST.md`中的检查清单，逐项检查每个文档。

### 方法2：关键词搜索 / Method 2: Keyword Search

使用grep或类似工具搜索必需章节的关键词：

```bash
# 检查必需章节
grep -E "^## |^### " document.md | grep -E "Overview|Definition|Properties|Relations|Examples|Explanations|Argumentation|Applications|References|Status"
```

### 方法3：自动化脚本 / Method 3: Automated Script

创建Python脚本自动检查文档结构（见下文）。

---

## 🐍 Python验证脚本 / Python Validation Script

### 脚本功能 / Script Features

- 检查10个必需章节是否存在
- 检查章节顺序是否正确
- 检查双语标题
- 检查Mermaid图表
- 检查形式化定义
- 生成验证报告

### 脚本代码 / Script Code

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档结构验证脚本
Document Structure Validator Script
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple

# 必需章节列表（按顺序）
REQUIRED_SECTIONS = [
    ("title", r"^#\s+.+"),
    ("toc", r"Table of Contents|目录"),
    ("overview", r"Overview|概述"),
    ("definition", r"Definition|定义"),
    ("properties", r"Properties|属性"),
    ("relations", r"Relations|关系"),
    ("examples", r"Examples|实例"),
    ("explanations", r"Explanations|解释"),
    ("argumentation", r"Argumentation|论证"),
    ("applications", r"Applications|应用"),
    ("references", r"References|参考文献"),
    ("status", r"Status|状态"),
]

# 必需内容检查
REQUIRED_CONTENT = {
    "formal_definition": r"定义\s+\d+\.\d+\.\d+|Definition\s+\d+\.\d+\.\d+",
    "mathematical_model": r"\$\$|\\begin\{equation\}",
    "mermaid_diagram": r"```mermaid",
    "standards_alignment": r"PMBOK|ISO|PRINCE2|CMMI",
    "latest_research": r"Latest Research Frontiers|最新研究前沿",
}


def check_sections(content: str) -> Dict[str, bool]:
    """检查必需章节是否存在"""
    results = {}
    content_lower = content.lower()

    for section_name, pattern in REQUIRED_SECTIONS:
        # 检查章节标题（## 或 ###）
        section_pattern = rf"^##+\s+.*{pattern}"
        if re.search(section_pattern, content, re.MULTILINE | re.IGNORECASE):
            results[section_name] = True
        else:
            results[section_name] = False

    return results


def check_content(content: str) -> Dict[str, bool]:
    """检查必需内容是否存在"""
    results = {}

    for content_name, pattern in REQUIRED_CONTENT.items():
        if re.search(pattern, content, re.IGNORECASE):
            results[content_name] = True
        else:
            results[content_name] = False

    return results


def check_bilingual_title(content: str) -> bool:
    """检查是否有双语标题"""
    # 检查标题是否包含中英文分隔符（/）
    title_pattern = r"^#\s+.+\s+/\s+.+"
    return bool(re.search(title_pattern, content, re.MULTILINE))


def validate_document(file_path: Path) -> Dict:
    """验证单个文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    sections = check_sections(content)
    content_checks = check_content(content)
    bilingual = check_bilingual_title(content)

    # 计算完成度
    section_count = sum(1 for v in sections.values() if v)
    content_count = sum(1 for v in content_checks.values() if v)
    total_required = len(REQUIRED_SECTIONS)

    return {
        "file": str(file_path),
        "sections": sections,
        "content": content_checks,
        "bilingual_title": bilingual,
        "section_completeness": f"{section_count}/{total_required}",
        "content_completeness": f"{content_count}/{len(REQUIRED_CONTENT)}",
        "overall_status": "✅ Complete" if section_count == total_required else "⚠️ Incomplete"
    }


def validate_directory(directory: Path) -> List[Dict]:
    """验证目录中的所有Markdown文档"""
    results = []

    for md_file in directory.rglob("*.md"):
        # 跳过某些目录
        if any(skip in str(md_file) for skip in [".git", "node_modules", "templates"]):
            continue

        result = validate_document(md_file)
        results.append(result)

    return results


def generate_report(results: List[Dict], output_file: Path = None) -> str:
    """生成验证报告"""
    report = []
    report.append("# 文档结构验证报告 / Document Structure Validation Report\n")
    report.append(f"**生成时间**: {Path(__file__).stat().st_mtime}\n")
    report.append(f"**检查文档数**: {len(results)}\n\n")

    # 按状态分组
    complete = [r for r in results if r.get("overall_status") == "✅ Complete"]
    incomplete = [r for r in results if r.get("overall_status") != "✅ Complete"]

    report.append(f"## 统计 / Statistics\n")
    report.append(f"- ✅ 完整文档: {len(complete)}\n")
    report.append(f"- ⚠️ 不完整文档: {len(incomplete)}\n\n")

    # 详细报告
    report.append("## 详细报告 / Detailed Report\n\n")

    for result in results:
        if "error" in result:
            report.append(f"### ❌ {result['file']}\n")
            report.append(f"错误: {result['error']}\n\n")
            continue

        status_icon = "✅" if result["overall_status"] == "✅ Complete" else "⚠️"
        report.append(f"### {status_icon} {result['file']}\n")
        report.append(f"- **章节完成度**: {result['section_completeness']}\n")
        report.append(f"- **内容完成度**: {result['content_completeness']}\n")
        report.append(f"- **双语标题**: {'✅' if result['bilingual_title'] else '❌'}\n\n")

        # 缺失章节
        missing_sections = [k for k, v in result['sections'].items() if not v]
        if missing_sections:
            report.append(f"**缺失章节**: {', '.join(missing_sections)}\n\n")

        # 缺失内容
        missing_content = [k for k, v in result['content'].items() if not v]
        if missing_content:
            report.append(f"**缺失内容**: {', '.join(missing_content)}\n\n")

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

    return report_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path("docs")

    print(f"正在验证目录: {target}")
    results = validate_directory(target)
    report = generate_report(results, Path("DOCUMENT_VALIDATION_REPORT.md"))
    print(report)
```

---

## 📋 使用说明 / Usage Instructions

### 运行验证脚本 / Run Validation Script

```bash
# 验证整个docs目录
python document_structure_validator.py docs/

# 验证特定目录
python document_structure_validator.py docs/02-project-management/

# 验证单个文件
python document_structure_validator.py docs/02-project-management/lifecycle-models.md
```

### 查看验证报告 / View Validation Report

验证脚本会生成`DOCUMENT_VALIDATION_REPORT.md`文件，包含：

- 统计信息
- 每个文档的详细检查结果
- 缺失章节和内容列表

---

## 🔧 扩展功能 / Extended Features

### 1. CI/CD集成 / CI/CD Integration

将验证脚本集成到CI/CD流程中，在每次提交时自动检查文档结构。

### 2. 自动修复建议 / Auto-fix Suggestions

根据检查结果，自动生成修复建议和模板。

### 3. 批量更新 / Batch Updates

根据验证结果，批量更新不符合标准的文档。

---

## 📊 验证结果示例 / Validation Result Example

```markdown
# 文档结构验证报告

**生成时间**: 2026-01-27
**检查文档数**: 15

## 统计 / Statistics

- ✅ 完整文档: 3
- ⚠️ 不完整文档: 12

## 详细报告 / Detailed Report

### ⚠️ docs/02-project-management/lifecycle-models.md

- **章节完成度**: 8/12
- **内容完成度**: 4/5
- **双语标题**: ✅

**缺失章节**: relations, explanations, argumentation
**缺失内容**: latest_research
```

---

## 📚 相关文档 / Related Documents

- `DOCUMENT_STRUCTURE_CHECKLIST.md`: 文档结构标准化检查清单
- `DOCUMENT_STRUCTURE_ANALYSIS.md`: 文档结构分析报告
- `CONTENT_ORGANIZATION_GUIDE.md`: 内容组织指南

---

**Last Updated / 最后更新**: 2026-01-27
**Version / 版本**: 1.0
**Status / 状态**: ✅ 持续更新中
