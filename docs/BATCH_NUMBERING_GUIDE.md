# 批量编号修复指南

## 概述

本文档提供批量修复文件内部编号的指南和脚本模板，用于统一所有文档的内部编号系统。

## 编号修复规则

### 规则1：标题编号与内部编号对应

如果文件标题是 `# X.Y`，那么：

- 一级子章节应该是 `## X.Y.1`, `## X.Y.2`, ...
- 二级子章节应该是 `### X.Y.Z.1`, `### X.Y.Z.2`, ...
- 定义/定理/算法编号应该是 `定义 X.Y.Z`, `定理 X.Y.Z`, ...

### 规则2：去除多余的层级

如果发现内部编号比标题多一层（如标题是 `4.2.2`，但内部是 `4.2.2.2.X`），需要去掉多余的层级。

## 编号映射表

### 软件开发模型 (4.1.X)

| 文件 | 标题编号 | 旧内部编号格式 | 新内部编号格式 |
|------|---------|--------------|--------------|
| agile-models.md | 4.1.1 | 4.2.1.1.X | 4.1.1.X |
| waterfall-models.md | 4.1.2 | 4.2.1.2.X | 4.1.2.X |
| spiral-models.md | 4.1.3 | 4.2.1.3.X | 4.1.3.X |
| iterative-models.md | 4.1.4 | 4.2.1.4.X | 4.1.4.X |
| devops-models.md | 4.1.5 | 4.2.1.5.X | 4.1.5.X |

### 工程管理模型 (4.2.X)

| 文件 | 标题编号 | 旧内部编号格式 | 新内部编号格式 |
|------|---------|--------------|--------------|
| systems-engineering.md | 4.2.1 | 4.2.2.1.X | 4.2.1.X |
| construction-engineering.md | 4.2.2 | 4.2.2.2.X | 4.2.2.X ✅ |
| mechanical-engineering.md | 4.2.3 | 4.2.2.3.X | 4.2.3.X |
| electrical-engineering.md | 4.2.4 | 4.2.2.4.X | 4.2.4.X |

### 商业管理模型 (4.3.X)

| 文件 | 标题编号 | 旧内部编号格式 | 新内部编号格式 |
|------|---------|--------------|--------------|
| strategic-management.md | 4.3.1 | 4.2.3.1.X | 4.3.1.X |
| operational-management.md | 4.3.2 | 4.2.3.2.X | 4.3.2.X |
| financial-management.md | 4.3.3 | 4.2.3.3.X | 4.3.3.X |
| human-resource-management.md | 4.3.4 | 4.2.3.4.X | 4.3.4.X |
| innovation-management.md | 4.3.5 | 4.2.4.1.X | 4.3.5.X |
| knowledge-management.md | 4.3.6 | 4.2.4.2.X | 4.3.6.X |
| change-management.md | 4.3.7 | 4.2.4.3.X | 4.3.7.X |

### 专业领域模型 (4.4.X)

| 文件 | 标题编号 | 旧内部编号格式 | 新内部编号格式 |
|------|---------|--------------|--------------|
| healthcare-management.md | 4.4.1 | 4.2.5.1.X | 4.4.1.X |
| education-management.md | 4.4.2 | 4.2.5.2.X | 4.4.2.X |
| fintech-management.md | 4.4.3 | 4.2.5.3.X | 4.4.3.X |
| logistics-management.md | 4.4.4 | 4.2.5.4.X | 4.4.4.X |
| energy-management.md | 4.4.5 | 4.2.5.5.X | 4.4.5.X |

### 新兴技术模型 (4.5.X)

| 文件 | 标题编号 | 旧内部编号格式 | 新内部编号格式 |
|------|---------|--------------|--------------|
| ai-management.md | 4.5.1 | 4.2.6.1.X | 4.5.1.X |
| blockchain-management.md | 4.5.2 | 4.2.6.2.X | 4.5.2.X |
| iot-management.md | 4.5.3 | 4.2.6.3.X | 4.5.3.X |
| quantum-management.md | 4.5.4 | 4.2.6.4.X | 4.5.4.X |

## 批量修复步骤

### 步骤1：识别需要修复的文件

使用以下命令查找需要修复的文件：

```bash
# 查找所有包含旧编号格式的文件
grep -r "4\.2\.\d\.\d\.\d" docs/04-industry-applications/
```

### 步骤2：逐个文件修复

对于每个文件：

1. **检查文件标题编号**

   ```bash
   head -1 <file>
   ```

2. **查找所有需要替换的编号**

   ```bash
   grep -n "4\.2\.\d\.\d\.\d" <file>
   ```

3. **执行批量替换**
   - 使用文本编辑器的查找替换功能
   - 或使用sed命令（需要小心处理）

### 步骤3：验证修复结果

修复后验证：

```bash
# 检查是否还有旧编号格式
grep -r "4\.2\.\d\.\d\.\d" docs/04-industry-applications/

# 检查编号一致性
grep -r "^## " docs/04-industry-applications/ | head -20
```

## 修复示例

### 示例1：agile-models.md

**修复前：**

```markdown
# 4.1.1 敏捷开发模型

## 4.2.1.1.1 概述
**定义 4.2.1.1.1** ...
```

**修复后：**

```markdown
# 4.1.1 敏捷开发模型

## 4.1.1.1 概述
**定义 4.1.1.1** ...
```

### 示例2：construction-engineering.md ✅

**修复前：**

```markdown
# 4.2.2 建筑工程模型

## 4.2.2.2.1 概述
**定义 4.2.2.2.1** ...
```

**修复后：**

```markdown
# 4.2.2 建筑工程模型

## 4.2.2.1 概述
**定义 4.2.2.1** ...
```

## 自动化脚本模板

### Python脚本示例

```python
import re
import os
from pathlib import Path

# 编号映射规则
NUMBERING_MAP = {
    '4.2.1.1': '4.1.1',
    '4.2.1.2': '4.1.2',
    '4.2.1.3': '4.1.3',
    '4.2.1.4': '4.1.4',
    '4.2.1.5': '4.1.5',
    '4.2.2.1': '4.2.1',
    '4.2.2.2': '4.2.2',
    '4.2.2.3': '4.2.3',
    '4.2.2.4': '4.2.4',
    '4.2.3.1': '4.3.1',
    '4.2.3.2': '4.3.2',
    '4.2.3.3': '4.3.3',
    '4.2.3.4': '4.3.4',
    '4.2.4.1': '4.3.5',
    '4.2.4.2': '4.3.6',
    '4.2.4.3': '4.3.7',
    '4.2.5.1': '4.4.1',
    '4.2.5.2': '4.4.2',
    '4.2.5.3': '4.4.3',
    '4.2.5.4': '4.4.4',
    '4.2.5.5': '4.4.5',
    '4.2.6.1': '4.5.1',
    '4.2.6.2': '4.5.2',
    '4.2.6.3': '4.5.3',
    '4.2.6.4': '4.5.4',
}

def fix_numbering_in_file(file_path):
    """修复文件中的编号"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 替换章节编号
    for old_pattern, new_pattern in NUMBERING_MAP.items():
        # 匹配 4.X.X.X.X 格式（去掉最后一层）
        pattern = old_pattern + r'\.(\d+)'
        replacement = new_pattern + r'.\1'
        content = re.sub(pattern, replacement, content)

        # 匹配 4.X.X.X 格式（直接替换）
        pattern = old_pattern + r'(?!\.)'
        content = re.sub(pattern, new_pattern, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# 遍历所有文件
docs_dir = Path('docs/04-industry-applications')
for md_file in docs_dir.rglob('*.md'):
    if fix_numbering_in_file(md_file):
        print(f"Fixed: {md_file}")
```

## 注意事项

1. **备份文件**：批量修改前务必备份所有文件
2. **逐步验证**：每次修改后验证结果
3. **引用链接**：修复编号后需要更新所有引用链接
4. **定义/定理编号**：确保定义、定理、算法等编号也同步更新

## 进度跟踪

### 已完成 ✅

- [x] construction-engineering.md (4.2.2)
- [x] agile-models.md (4.1.1) - 部分完成

### 待完成 ⏳

- [ ] 其他23个行业应用模型文件
- [ ] 所有文件的引用链接更新
- [ ] 交叉引用验证

---

**最后更新**：2025-01-XX
**维护者**：Formal-ProgramManage团队
