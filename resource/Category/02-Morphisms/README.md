# Morphisms / 态射

## 📋 Overview / 概述

This directory contains **Formal-ProgramManage** project management operations and transformations organized as **morphisms** in categories.

本目录包含作为范畴中的**态射**组织的**Formal-ProgramManage**项目管理运算和变换。

## 🔗 与 docs 的层、转换对应 / Layer and Transformation

**态射 = 转换**：

- **生命周期转换** $\delta$：`08-Lifecycle-Morphisms.md` 对应 docs/02-project-management/lifecycle-models 的 $\delta: S \times \Sigma \rightarrow S$、转换点 $T$
- **状态转换** $\rightarrow$：`01-Formal-Morphisms.md`、`02-Mathematical-Morphisms.md`、`03-Semantic-Morphisms.md` 对应 docs/01-foundations 的 Kripke 结构、语义模型中的 $\rightarrow \subseteq S \times S$
- **资源/风险/质量转换**：`09-Resource-Morphisms.md`、`10-Risk-Morphisms.md`、`11-Quality-Morphisms.md` 对应核心模型层的状态转换
- **模型/等价转换**：`12-Verification-Morphisms.md`、`13-Proof-Morphisms.md`、`14-Consistency-Morphisms.md` 对应 docs/06-ci-verification 的模型检验、一致性检查、定理证明；**13–22 已补「与 docs 的公式对应」**

## 📁 Files / 文件（25 文件，含 0. 所属层与转换关系）

**注意**：Quantum、Biological、Holographic、Interstellar、Energy、Network、System 共 7 个已归档至 `_archive/02-Morphisms/`。

**项目管理核心态射（态射 = 转换）**：

- `08-Lifecycle-Morphisms.md` - Lifecycle morphisms / 生命周期态射（对应生命周期转换 $\delta$）
- `09-Resource-Morphisms.md` - Resource morphisms / 资源态射
- `10-Risk-Morphisms.md` - Risk morphisms / 风险态射
- `11-Quality-Morphisms.md` - Quality morphisms / 质量态射
- `12-Verification-Morphisms.md` - Verification morphisms / 验证态射
- `14-Consistency-Morphisms.md` - Consistency morphisms / 一致性态射（对应模型/等价转换）

**形式/数学/语义态射（对应状态转换 $\rightarrow$）**：

- `01-Formal-Morphisms.md` - Formal morphisms / 形式态射
- `02-Mathematical-Morphisms.md` - Mathematical morphisms / 数学态射
- `03-Semantic-Morphisms.md` - Semantic morphisms / 语义态射

**程序分析态射**：

- `15-Control-Morphisms.md` - Control morphisms / 控制态射
- `16-Dataflow-Morphisms.md` - Dataflow morphisms / 数据流态射
- `17-Execution-Morphisms.md` - Execution morphisms / 执行态射

**证明/语义/替换与行业态射**：`13-Proof-Morphisms`、`18-Scope-Morphisms`、`19-Substitution-Morphisms`、`20-Denotational-Semantics-Morphisms`、`21-Axiomatic-Semantics-Morphisms`、`22-Replacement-Morphisms`；行业 `30-Construction`～`36-AI-Morphisms`。

**说明**：各 .md 均已补充「0. 所属层与转换关系」。

**注意**：微积分相关态射（Differentiation、Integration、Laplace、Fourier、Function-Composition）已归档至 `_archive/`，详见 [ARCHIVE_LIST.md](../../ARCHIVE_LIST.md)

## 🔗 Alignment / 对齐

**From Concept / 从概念（项目管理向）**:

- `resource/Concept/02-生命周期概念/` → Lifecycle morphisms（生命周期转换 $\delta$）
- `resource/Concept/03-资源管理概念/` → Resource morphisms
- `resource/Concept/04-风险管理概念/` → Risk morphisms
- `resource/Concept/05-质量管理概念/` → Quality morphisms
- `resource/Concept/07-程序分析概念/` → Verification morphisms（模型转换）

**From Transfer / 从变换（项目管理向）**:

- `resource/Transfer/01-等价关系框架/` → Verification、Consistency morphisms（模型/等价转换）
- `resource/Transfer/02-变换类型框架/` → Lifecycle morphisms（生命周期转换 $\delta$）

## 📚 Key Concepts / 关键概念

### Project Management Morphisms / 项目管理态射（态射 = 转换）

- **Lifecycle Morphisms**: Phase transitions $\tau: \mathbf{Phase}_i \to \mathbf{Phase}_{i+1}$（对应生命周期转换 $\delta$）
- **Resource/Risk/Quality Morphisms**: Allocation, identification, response, assurance, control operations
- **Verification Morphisms**: Model checking, consistency checking（对应模型/等价转换）

---

**Status / 状态**: ✅ Complete / 完成
**Last Updated / 最后更新**: 2026-01-27
