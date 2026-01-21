# Categories / 范畴

## 📋 Overview / 概述

This directory contains descriptions of specific categories for **Formal-ProgramManage** project management.

本目录包含**Formal-ProgramManage**项目管理的具体范畴的描述。

## 🔗 与 docs 的层、转换对应 / Layer and Transformation

- **项目管理范畴**：Project、Lifecycle、Resource、Risk、Quality 等范畴对应各层的实体与转换
- **程序分析范畴**：Type、Control、Data、Execution 等范畴支撑形式化验证（docs/06-ci-verification）

## 📁 Files / 文件（4 文件）

**程序分析范畴**（支撑形式化验证，对应 docs/06-ci-verification）：

- `01-Control-Category.md` - Control category / 控制范畴
- `02-Data-Flow-Category.md` - Data flow category / 数据流范畴
- `03-Execution-Category.md` - Execution category / 执行范畴
- `04-Type-Category.md` - Type category / 类型范畴

**注意**：微积分相关范畴（Func、Diff、Integrable，即 `01-Func-Category.md`、`02-Diff-Category.md`、`03-Integrable-Category.md`）已归档至 `_archive/`，详见 [ARCHIVE_LIST.md](../../ARCHIVE_LIST.md)

## 🔗 Alignment / 对齐

### From Concept / 从概念（项目管理向）

- `resource/Concept/01-项目管理基础/` → Project category
- `resource/Concept/02-生命周期概念/` → Lifecycle category
- `resource/Concept/03-资源管理概念/` → Resource category
- `resource/Concept/04-风险管理概念/` → Risk category
- `resource/Concept/05-质量管理概念/` → Quality category
- `resource/Concept/06-编程语言理论概念/` → Type、Control、Data、Execution categories

### Cross-References / 交叉引用

- **Foundations**: See `00-Foundations/` for category definitions（02-Calculus-Categories 已归档）
- **Objects**: See `01-Objects/` for objects in these categories
- **Morphisms**: See `02-Morphisms/` for morphisms in these categories（态射 = 转换）
- **Functors**: See `04-Functors/` for functors between categories（函子 = 层间映射）
- **Constructions**: See `03-Constructions/` for universal constructions in these categories

## 📚 Key Concepts / 关键概念

### Project Management Categories / 项目管理范畴

- **Project Category**: Objects are projects, subprojects, project states; morphisms are state transitions, phase transitions（对应生命周期转换 $\delta$、状态转换 $\rightarrow$）
- **Lifecycle Category**: Objects are phases (initiation, planning, execution, monitoring, closure); morphisms are lifecycle transitions（对应生命周期转换 $\delta$）
- **Resource/Risk/Quality Categories**: Objects are resource/risk/quality entities; morphisms are allocation, identification, response, assurance, control operations

### Program Analysis Categories / 程序分析范畴

- **Type Category**: Objects are types; morphisms are type operations（支撑形式化验证）
- **Control/Data/Execution Categories**: Objects are control/data/execution states; morphisms are state transitions（支撑程序分析）

---

**Status / 状态**: ✅ Complete / 完成
**Last Updated / 最后更新**: 2026-01-27
