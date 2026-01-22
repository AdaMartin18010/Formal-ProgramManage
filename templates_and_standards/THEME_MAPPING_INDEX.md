# Formal-ProgramManage项目主题映射索引

## 文档说明

本文档提供Formal-ProgramManage（正式程序管理）项目主题的完整映射索引，包括：

- 项目结构到主题的映射
- 主题到国际标准的映射
- 主题代码索引
- 快速查找指南

所有映射对齐PMI PMBOK 7th Edition、ISO 21500:2012、ISO 31000:2018、PRINCE2 2017、CMMI-DEV等国际项目管理标准。

**数据来源**: 基于docs目录中所有文件的内容梳理

**创建时间**: 2026-01-27
**最后更新**: 2026-01-27
**版本**: 2.0
**状态**: ✅ 已完成项目管理主题映射索引创建

---

## 一、主题代码索引

### 1.1 层次代码

- **FL**: Foundation Layer (基础理论层)
- **CML**: Core Model Layer (核心模型层)
- **VL**: Verification Layer (验证理论层)
- **AL**: Application Layer (应用模型层)
- **IL**: Implementation Layer (实现验证层)

### 1.2 主题代码格式

```text
[层次代码]-[二级主题编号].[三级主题编号].[四级主题编号]
```

**示例**:

- `FL-1.1-1.1.1` = 基础理论层-形式化基础理论-项目定义与形式化规范-项目四元组定义
- `CML-2.1-2.1.1` = 核心模型层-项目生命周期模型-启动阶段-项目章程
- `VL-3.1-3.1.1` = 验证理论层-形式化验证理论-模型检验-状态空间搜索
- `AL-4.1-4.1.1` = 应用模型层-软件开发模型-敏捷开发模型-Scrum
- `IL-5.1-5.1.1` = 实现验证层-Rust实现-项目状态转换系统

---

## 二、项目结构到主题映射

### 2.1 基础理论层 (FL) → docs/01-foundations/

| 文件 | 主题代码 | 主题名称 | 标准对标 |
|------|---------|---------|---------|
| README.md | FL-1.0 | 形式化基础理论概述 | 形式化方法标准 |
| mathematical-models.md | FL-1.2 | 数学模型基础 | 数学标准 |
| semantic-models.md | FL-1.3 | 语义模型理论 | 形式化方法标准 |
| quantum-project-theory.md | FL-1.4 | 量子项目管理理论 | 量子计算标准 |
| bio-inspired-project-theory.md | FL-1.5 | 生物启发式项目管理理论 | 生物计算标准 |
| holographic-project-theory.md | FL-1.6 | 全息项目管理理论 | 全息理论标准 |
| interstellar-project-theory.md | FL-1.7 | 星际项目管理理论 | 航天项目管理标准 |

### 2.2 核心模型层 (CML) → docs/02-project-management/

| 文件 | 主题代码 | 主题名称 | 标准对标 |
|------|---------|---------|---------|
| README.md | CML-2.0 | 项目管理核心模型概述 | PMBOK 7th Edition、ISO 21500:2012 |
| lifecycle-models.md | CML-2.1 | 项目生命周期模型 | PMBOK 7th Edition、ISO 21500:2012、PRINCE2 2017 |
| resource-models.md | CML-2.2 | 资源管理模型 | PMBOK 7th Edition、ISO 21500:2012 |
| risk-models.md | CML-2.3 | 风险管理模型 | PMBOK 7th Edition、ISO 31000:2018 |
| quality-models.md | CML-2.4 | 质量管理模型 | ISO/IEC 25010、ISO 9001、CMMI-DEV |

### 2.3 验证理论层 (VL) → docs/03-formal-verification/

| 文件 | 主题代码 | 主题名称 | 标准对标 |
|------|---------|---------|---------|
| verification-theory.md | VL-3.1 | 形式化验证理论 | Model Checking、Theorem Proving |
| model-checking.md | VL-3.2 | 模型检验方法 | Model Checking标准 |
| theorem-proving.md | VL-3.3 | 定理证明系统 | Theorem Proving标准 |

### 2.4 应用模型层 (AL) → docs/04-industry-applications/

| 目录/文件 | 主题代码 | 主题名称 | 标准对标 |
|----------|---------|---------|---------|
| software-development/ | AL-4.1 | 软件开发模型 | Agile、DevOps标准 |
| engineering-management/ | AL-4.2 | 工程管理模型 | 工程管理标准 |
| business-management/ | AL-4.3 | 商业管理模型 | 商业管理标准 |
| ai-management/ | AL-4.4 | AI管理模型 | AI项目管理标准 |
| blockchain-management/ | AL-4.5 | 区块链管理模型 | 区块链标准 |
| iot-management/ | AL-4.6 | IoT管理模型 | IoT标准 |
| quantum-management/ | AL-4.7 | 量子管理模型 | 量子计算标准 |
| healthcare-management/ | AL-4.8 | 医疗管理模型 | 医疗项目管理标准 |
| education-management/ | AL-4.9 | 教育管理模型 | 教育项目管理标准 |
| fintech-management/ | AL-4.10 | 金融管理模型 | 金融项目管理标准 |
| logistics-management/ | AL-4.11 | 物流管理模型 | 物流项目管理标准 |
| energy-management/ | AL-4.12 | 能源管理模型 | 能源项目管理标准 |

### 2.5 实现验证层 (IL) → docs/05-implementations/

| 文件 | 主题代码 | 主题名称 | 标准对标 |
|------|---------|---------|---------|
| rust-examples.md | IL-5.1 | Rust实现 | Rust标准 |
| haskell-examples.md | IL-5.2 | Haskell实现 | Haskell标准 |
| lean-examples.md | IL-5.3 | Lean实现 | Lean标准 |

---

## 三、主题到国际标准映射

### 3.1 PMBOK 7th Edition映射

| PMBOK组件 | 对应主题层次 | 主题代码 |
|----------|------------|---------|
| 12个项目管理原则 | CML-2.1 ~ CML-2.4 | 核心模型层 |
| 8个绩效域 | CML-2.1 ~ CML-2.4 | 核心模型层 |
| 价值交付系统 | CML-2.1 | CML-2.1 |
| 模型、方法和工件 | AL-4.1 ~ AL-4.12 | 应用模型层 |

### 3.2 ISO 21500:2012映射

| ISO组件 | 对应主题层次 | 主题代码 |
|---------|------------|---------|
| 39个项目管理过程 | CML-2.1 | CML-2.1.1 ~ CML-2.1.5 |
| 10个知识领域 | CML-2.1 ~ CML-2.4 | CML-2.1 ~ CML-2.4 |
| 5个过程组 | CML-2.1 | CML-2.1.1 ~ CML-2.1.5 |

### 3.3 ISO 31000:2018映射

| ISO组件 | 对应主题层次 | 主题代码 |
|---------|------------|---------|
| 风险管理框架 | CML-2.3 | CML-2.3 |
| 风险管理过程 | CML-2.3 | CML-2.3.1 ~ CML-2.3.4 |

### 3.4 PRINCE2 2017映射

| PRINCE2组件 | 对应主题层次 | 主题代码 |
|------------|------------|---------|
| 7个原则 | CML-2.1 ~ CML-2.4 | 核心模型层 |
| 7个主题 | CML-2.1 ~ CML-2.4 | 核心模型层 |
| 7个过程 | CML-2.1 | CML-2.1.1 ~ CML-2.1.5 |

### 3.5 CMMI-DEV映射

| CMMI组件 | 对应主题层次 | 主题代码 |
|---------|------------|---------|
| 22个过程域 | CML-2.1 ~ CML-2.4, AL-4.1 | 核心模型层、应用模型层 |
| 5个成熟度等级 | CML-2.1 ~ CML-2.4 | 核心模型层 |

---

## 四、核心概念映射

### 4.1 基础概念映射

| 概念 | 文件位置 | 主题代码 | 标准对标 |
|------|---------|---------|---------|
| 项目 | docs/01-foundations/README.md | FL-1.1-1.1.1 | ISO 21500:2012 |
| 项目管理 | docs/01-foundations/README.md | FL-1.1-1.1.1 | PMBOK 7th Edition |
| 项目生命周期 | docs/02-project-management/lifecycle-models.md | CML-2.1 | PMBOK 7th Edition、ISO 21500:2012 |
| 资源管理 | docs/02-project-management/resource-models.md | CML-2.2 | PMBOK 7th Edition |
| 风险管理 | docs/02-project-management/risk-models.md | CML-2.3 | PMBOK 7th Edition、ISO 31000:2018 |
| 质量管理 | docs/02-project-management/quality-models.md | CML-2.4 | ISO/IEC 25010、ISO 9001 |

### 4.2 形式化概念映射

| 概念 | 文件位置 | 主题代码 | 标准对标 |
|------|---------|---------|---------|
| 状态转换系统 | docs/01-foundations/README.md | FL-1.1-1.1.1 | Model Checking标准 |
| 时序逻辑 | docs/01-foundations/README.md | FL-1.1-1.1.2 | Temporal Logic标准 |
| 模型检验 | docs/03-formal-verification/model-checking.md | VL-3.2 | Model Checking标准 |
| 定理证明 | docs/03-formal-verification/theorem-proving.md | VL-3.3 | Theorem Proving标准 |

---

## 五、快速查找指南

### 5.1 按主题代码查找

1. 确定层次代码（FL/CML/VL/AL/IL）
2. 确定二级主题编号（1.1-5.3）
3. 确定三级主题编号（如需要）
4. 在映射表中查找对应的项目路径和标准对标

### 5.2 按项目结构查找

1. 确定文档目录（docs/01-foundations/等）
2. 确定文件名称
3. 在"项目结构到主题映射"表中查找主题代码
4. 在"主题到国际标准映射"表中查找标准对标

### 5.3 按国际标准查找

1. 确定国际标准（PMBOK、ISO、PRINCE2、CMMI等）
2. 确定标准组件
3. 在"主题到国际标准映射"表中查找主题代码
4. 在"项目结构到主题映射"表中查找项目路径

### 5.4 按概念名称查找

1. 在"核心概念映射"表中查找概念
2. 获取文件位置和主题代码
3. 在"项目结构到主题映射"表中查找项目路径
4. 在"主题到国际标准映射"表中查找标准对标

---

## 六、相关文档

- `THEME_HIERARCHY_MASTER.md`: 主题层次结构主文档
- `THEME_HIERARCHY_COMPLETE.md`: 完整主题层次结构
- `THEME_CLASSIFICATION_STANDARD.md`: 主题分类标准
- `THEME_CLASSIFICATION_COMPLETE.md`: 完整主题分类标准
- `CONTENT_ORGANIZATION_GUIDE.md`: 内容组织指南
- `PROJECT_STRUCTURE.md`: 项目结构规划
- `术语表-Glossary.md`: 项目管理术语表

---

**文档创建时间**: 2026-01-27
**最后更新**: 2026-01-27
**版本**: 2.0
**状态**: ✅ 已完成项目管理主题映射索引创建
