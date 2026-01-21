# Risk Objects / 风险对象

## 📋 Table of Contents / 目录

- [Risk Objects / 风险对象](#risk-objects--风险对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Risks / 风险范畴](#21-category-of-risks--风险范畴)
    - [2.2 Risk Object Properties / 风险对象性质](#22-risk-object-properties--风险对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义](#31-pmbok-7th-edition-definition--pmbok-第7版定义)
    - [3.2 ISO 31000 Standard Definition / ISO 31000 标准定义](#32-iso-31000-standard-definition--iso-31000-标准定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Technical Risk Example / 技术风险例子](#61-technical-risk-example--技术风险例子)
    - [6.2 Schedule Risk Example / 进度风险例子](#62-schedule-risk-example--进度风险例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
    - [7.2 Category-Theoretic Applications / 范畴论应用](#72-category-theoretic-applications--范畴论应用)
    - [7.3 直观解释与一例 / Intuitive Explanation with One Example](#73-直观解释与一例--intuitive-explanation-with-one-example)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Standards / 标准](#81-standards--标准)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**核心模型层**（对应 docs/02-project-management；风险管理模型）
- **转换关系**：**Risk Objects** 作为**状态转换**的实体（风险识别、分析、应对作为状态转换）；与 04-风险管理概念、Category/02-Morphisms/10-Risk-Morphisms、Category/04-Functors/03-Risk-Management-Functor 对应。

**与 docs 的公式对应**：风险五元组 $\mathcal{R}=(E,P,I,T,C)$、$\mathrm{Exposure}(e)=P(e)\times I(e)$、$\mathrm{Priority}(e)=\mathrm{rank}(\mathrm{Exposure}(e))$、分类 $\mathcal{C}=\{C_1,\ldots,C_k\}$ 见 `docs/02-project-management/risk-models`。

---

## 1. Overview / 概述

**English / 英文**:

Risk objects represent project risks (events, states, responses) in the category $\mathbf{Risk}$. They capture risk identification, analysis, response planning, and monitoring in project management. This document provides a category-theoretic perspective on risk objects, aligning with PMBOK 7th Edition and ISO 31000 standards.

**中文**:

风险对象表示项目范畴 $\mathbf{Risk}$ 中的项目风险（事件、状态、应对）。它们捕捉项目管理中的风险识别、分析、应对规划和监控。本文档从范畴论视角提供风险对象的定义，对齐 PMBOK 第7版和 ISO 31000 标准。

**Key Insights / 关键洞察**:

- **Risk Events / 风险事件**: Uncertain events that may affect project objectives / 可能影响项目目标的不确定事件
- **Risk States / 风险状态**: Current state of risks (identified, analyzed, responded) / 风险的当前状态（已识别、已分析、已应对）
- **Risk Responses / 风险应对**: Strategies to address risks / 应对风险的策略
- **Risk Assessment / 风险评估**: Probability and impact analysis / 概率和影响分析

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Risks / 风险范畴

**Definition 2.1** (Category $\mathbf{Risk}$)

The category $\mathbf{Risk}$ is defined as follows:

- **Objects / 对象**: Risks $Risk = (Event, Probability, Impact, State, Response)$ where:
  - $Event$ is the risk event description
  - $Probability \in [0,1]$ is the probability of occurrence
  - $Impact \in \mathbb{R}^+$ is the impact magnitude
  - $State \in \{Identified, Analyzed, Responded, Monitored\}$ is the risk state
  - $Response \in \{Avoid, Mitigate, Transfer, Accept\}$ is the response strategy

- **Morphisms / 态射**: Risk transformations $f: Risk_1 \to Risk_2$ representing risk evolution

- **Composition / 复合**: Composition of risk transformations $(g \circ f): Risk_1 \to Risk_3$

- **Identity / 恒等**: Identity transformation $\text{id}_{Risk}: Risk \to Risk$

### 2.2 Risk Object Properties / 风险对象性质

**Axiom 2.1** (Risk Probability Boundedness)

For any risk $Risk = (Event, Probability, Impact, State, Response)$:
$$0 \leq Probability \leq 1$$

**Axiom 2.2** (Risk Impact Non-negativity)

For any risk $Risk$:
$$Impact \geq 0$$

**Axiom 2.3** (Risk Exposure)

Risk exposure is defined as:
$$Exposure(Risk) = Probability \times Impact$$

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Project Risk - PMBOK 7th Edition)

A project risk is an uncertain event or condition that, if it occurs, has a positive or negative effect on one or more project objectives. In our formalization:

$$Risk(P) = \{Risk_1, Risk_2, \ldots, Risk_n\}$$

where each $Risk_i$ is an object in $\mathbf{Risk}$.

**Risk Categories / 风险类别**:

- **Threats / 威胁**: Negative risks that may harm the project
- **Opportunities / 机会**: Positive risks that may benefit the project

### 3.2 ISO 31000 Standard Definition / ISO 31000 标准定义

**Definition 3.2** (Risk - ISO 31000:2018)

Risk is the effect of uncertainty on objectives. In our category-theoretic framework:

$$Risk: \mathbf{Project} \to \mathbf{Risk}$$

where $Risk$ is the risk management functor.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Risk Identification Function)

Risk identification is a function:
$$identify: \mathbf{Project} \to 2^{\mathbf{Risk}}$$

**Property 4.2** (Risk Analysis Function)

Risk analysis computes probability and impact:
$$analyze: \mathbf{Risk} \to [0,1] \times \mathbb{R}^+$$

**Property 4.3** (Risk Response Function)

Risk response assigns strategies:
$$respond: \mathbf{Risk} \to \mathbf{Response}$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Risk Management Functor)

Risk management is a functor:
$$Risk: \mathbf{Project} \to \mathbf{Risk}$$

**Property 4.5** (Risk Composition)

Risks compose under aggregation:
$$aggregate(Risk_1, Risk_2) = (Event_1 \cup Event_2, \max(Probability_1, Probability_2), Impact_1 + Impact_2, \ldots)$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Risk → Project)

Every risk belongs to a project:
$$Risk^{-1}: \mathbf{Risk} \to \mathbf{Project}$$

**Relation 5.2** (Risk → Resources)

Risks affect resource requirements:
$$R \circ Risk: \mathbf{Project} \to \mathbf{Resource}$$

**Relation 5.3** (Risk → Quality)

Risks impact quality attributes:
$$Q \circ Risk: \mathbf{Project} \to \mathbf{Quality}$$

**Relation 5.4** (Risk → Lifecycle)

Risks are associated with lifecycle phases:
$$L \circ Risk: \mathbf{Project} \to \mathbf{Phase} \times \mathbf{Risk}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Resource-Risk)

There exists a natural transformation $\beta: R \Rightarrow Risk$:
$$\beta_P: R(P) \to Risk(P)$$

connecting resource constraints to risks.

**Natural Transformation 5.2** (Risk-Quality)

There exists a natural transformation $\gamma: Risk \Rightarrow Q$:
$$\gamma_P: Risk(P) \to Q(P)$$

connecting risk impacts to quality.

---

## 6. Examples / 例子

### 6.1 Technical Risk Example / 技术风险例子

**Example 6.1** (Technology Risk)

Consider a technology risk:

$$Risk_{tech} = (Event_{tech}, 0.3, 50k, Analyzed, Mitigate)$$

where:

- $Event_{tech}$: "New technology may not meet performance requirements"
- $Probability = 0.3$ (30% chance)
- $Impact = \$50k$ (cost impact)
- $State = Analyzed$
- $Response = Mitigate$ (mitigation strategy)

### 6.2 Schedule Risk Example / 进度风险例子

**Example 6.2** (Schedule Risk)

Consider a schedule risk:

$$Risk_{schedule} = (Event_{schedule}, 0.5, 30 \text{ days}, Identified, Transfer)$$

where:

- $Event_{schedule}$: "Key resource may be unavailable"
- $Probability = 0.5$ (50% chance)
- $Impact = 30$ days delay
- $State = Identified$
- $Response = Transfer$ (transfer to vendor)

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Risk Identification**: Identifying project risks
- **Risk Analysis**: Analyzing risk probability and impact
- **Risk Response Planning**: Planning risk responses
- **Risk Monitoring**: Monitoring risk status

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Risk Composition**: Composing risks using morphisms
- **Risk Transformation**: Transforming risks using functors
- **Risk Relationships**: Understanding relationships using natural transformations

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Risk as a Project-Derived Set / 风险即由项目导出的集合)

风险对象 $Rk=(Events,Prob,Impact,Mit)$ 表示**风险集**：事件、概率、影响、缓解。函子 $Risk:\mathbf{Project}\to\mathbf{Risk}$ 给出 $Risk(P)$。例：$Risk(P_{sw})=\{\text{技术债},\text{需求变更},\text{关键人离职},\ldots\}$，每个 $r$ 带有 $(p,i)$ 与缓解 $Mit(r)$。态射 $f:Risk(P_1)\to Risk(P_2)$ 可表示风险传递或合并。自然变换 $\beta: R\Rightarrow Risk$ 体现资源紧张与风险升高之间的关联。

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 31000:2018. Risk management - Guidelines. International Organization for Standardization.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Project Objects](01-Project-Objects.md)
- [Resource Objects](09-Resource-Objects.md)
- [Risk Morphisms](../../02-Morphisms/10-Risk-Morphisms.md)
- [Risk Management Functor](../../04-Functors/03-Risk-Management-Functor.md)
- **docs**：`docs/02-project-management/risk-models`（$\mathcal{R}=(E,P,I,T,C)$、Exposure、Priority、分类 $\mathcal{C}$；与 0. 公式对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
