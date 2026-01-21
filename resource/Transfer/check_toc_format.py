#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOC格式验证脚本 / TOC Format Verification Script

检查Transfer目录中所有Markdown文件的TOC格式统一性。
"""

import os
import re
from pathlib import Path

def check_toc_format(file_path):
    """检查文件是否有标准TOC格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 检查是否有标准TOC
            has_standard_toc = bool(re.search(r'^## 📋 Table of Contents / 目录', content, re.MULTILINE))
            return has_standard_toc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def verify_transfer_directory():
    """验证Transfer目录所有文件的TOC格式"""
    transfer_dir = Path('resource/Transfer')
    md_files = list(transfer_dir.rglob('*.md'))
    
    # 排除README.md和INDEX.md等索引文件
    md_files = [f for f in md_files if f.name not in ['README.md', 'INDEX.md', 'TOC_VERIFICATION_SCRIPT.md']]
    
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
            print(f"  - {f.relative_to(Path('resource/Transfer'))}")
        if len(files_without_toc) > 20:
            print(f"  ... 还有 {len(files_without_toc) - 20} 个文件")
    
    return files_without_toc

if __name__ == "__main__":
    verify_transfer_directory()
