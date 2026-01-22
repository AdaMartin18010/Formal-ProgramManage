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
from datetime import datetime

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
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**检查文档数**: {len(results)}\n\n")
    
    # 按状态分组
    complete = [r for r in results if r.get("overall_status") == "✅ Complete"]
    incomplete = [r for r in results if r.get("overall_status") != "✅ Complete"]
    errors = [r for r in results if "error" in r]
    
    report.append(f"## 统计 / Statistics\n")
    report.append(f"- ✅ 完整文档: {len(complete)}\n")
    report.append(f"- ⚠️ 不完整文档: {len(incomplete)}\n")
    if errors:
        report.append(f"- ❌ 错误文档: {len(errors)}\n")
    report.append("\n")
    
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
        print(f"报告已保存到: {output_file}")
    
    return report_text


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path("docs")
    
    if not target.exists():
        print(f"错误: 路径不存在: {target}")
        sys.exit(1)
    
    print(f"正在验证: {target}")
    
    if target.is_file():
        results = [validate_document(target)]
    else:
        results = validate_directory(target)
    
    output_file = Path("DOCUMENT_VALIDATION_REPORT.md")
    report = generate_report(results, output_file)
    print("\n" + "="*50)
    print(report)
