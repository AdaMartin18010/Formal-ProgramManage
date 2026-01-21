#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept目录TOC格式验证脚本 / Concept Directory TOC Format Verification Script

检查Concept目录中支持资源文件的TOC格式统一性。
"""

import os
import re
from pathlib import Path

def check_toc_format(file_path):
    """检查文件是否有标准TOC格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 检查是否有标准TOC（多种可能的格式）
            has_standard_toc = (
                bool(re.search(r'^## 📋 Table of Contents / 目录', content, re.MULTILINE)) or
                bool(re.search(r'^## Table of Contents', content, re.MULTILINE)) or
                bool(re.search(r'^## 目录', content, re.MULTILINE))
            )
            return has_standard_toc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def verify_concept_directory():
    """验证Concept目录支持资源文件的TOC格式"""
    concept_dir = Path('resource/Concept')
    
    # 查找支持资源文件（快速参考、概念总结等）
    support_dirs = [
        '快速参考',
        '概念总结',
        '应用案例',  # 应用案例已经统一
    ]
    
    md_files = []
    for subdir in concept_dir.iterdir():
        if subdir.is_dir():
            # 查找所有子目录中的md文件
            md_files.extend(subdir.rglob('*.md'))
    
    # 排除主概念文件（通常已经有TOC）
    # 主要关注支持资源文件
    md_files = [f for f in md_files if f.name not in ['README.md', 'INDEX.md']]
    
    files_with_toc = []
    files_without_toc = []
    
    for md_file in md_files:
        if check_toc_format(md_file):
            files_with_toc.append(md_file)
        else:
            files_without_toc.append(md_file)
    
    print(f"总文件数: {len(md_files)}")
    print(f"有标准TOC的文件: {len(files_with_toc)}")
    print(f"没有标准TOC的文件: {len(files_without_toc)}")
    
    if files_without_toc:
        print(f"\n需要添加TOC的文件 (前20个):")
        for f in files_without_toc[:20]:
            print(f"  - {f.relative_to(Path('resource/Concept'))}")
        if len(files_without_toc) > 20:
            print(f"  ... 还有 {len(files_without_toc) - 20} 个文件")
    
    return files_without_toc

if __name__ == "__main__":
    verify_concept_directory()
