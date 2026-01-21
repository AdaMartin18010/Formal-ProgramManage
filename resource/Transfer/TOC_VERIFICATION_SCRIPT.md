# TOC格式验证脚本说明 / TOC Format Verification Script

## 📋 目的 / Purpose

此文档说明如何验证Transfer目录中所有文件的TOC格式统一性。

This document explains how to verify TOC format consistency across all files in the Transfer directory.

## ✅ 标准TOC格式 / Standard TOC Format

所有文件应使用以下标准TOC格式：

All files should use the following standard TOC format:

```markdown
## 📋 Table of Contents / 目录

- [Title / 标题](#title--标题)
  - [Section 1 / 章节1](#section-1--章节1)
  - [Section 2 / 章节2](#section-2--章节2)
```

## 🔍 验证方法 / Verification Methods

### 方法1：使用grep查找

```bash
# 查找有标准TOC的文件
grep -r "^## 📋 Table of Contents" resource/Transfer --include="*.md" | wc -l

# 查找没有标准TOC的文件（需要手动检查）
# 统计总文件数
find resource/Transfer -name "*.md" -type f | wc -l
```

### 方法2：Python脚本验证

```python
import os
import re
from pathlib import Path

def check_toc_format(file_path):
    """检查文件是否有标准TOC格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 检查是否有标准TOC
        has_standard_toc = bool(re.search(r'^## 📋 Table of Contents / 目录', content, re.MULTILINE))
        return has_standard_toc

def verify_transfer_directory():
    """验证Transfer目录所有文件的TOC格式"""
    transfer_dir = Path('resource/Transfer')
    md_files = list(transfer_dir.rglob('*.md'))

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
        print("\n需要添加TOC的文件:")
        for f in files_without_toc:
            print(f"  - {f}")

    return files_without_toc

if __name__ == "__main__":
    verify_transfer_directory()
```

## 📊 当前状态 / Current Status

根据之前的检查：

- ✅ 约176个文件已有标准TOC格式
- ⚠️ 剩余约43-63个文件待处理（估计值）

## 🎯 下一步行动 / Next Actions

1. 运行验证脚本识别需要更新的文件
2. 批量更新这些文件的TOC格式
3. 验证所有锚点链接有效性
4. 创建最终验证报告

---

**创建日期**: 2026-01-27
**状态**: 📋 待执行
