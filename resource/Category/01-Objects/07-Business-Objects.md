# Business Objects / 商业对象

## 📋 Table of Contents / 目录

- [Business Objects / 商业对象](#business-objects--商业对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Business / 商业范畴](#21-category-of-business--商业范畴)
    - [2.2 Business Object Properties / 商业对象性质](#22-business-object-properties--商业对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Business Project Definition / 商业项目定义](#31-business-project-definition--商业项目定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Product Launch Example / 产品发布例子](#61-product-launch-example--产品发布例子)
    - [6.2 Market Expansion Example / 市场扩张例子](#62-market-expansion-example--市场扩张例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
    - [7.3 直观解释与一例 / Intuitive Explanation with One Example](#73-直观解释与一例--intuitive-explanation-with-one-example)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Standards / 标准](#81-standards--标准)
    - [8.2 Related Files / 相关文件](#82-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**应用模型层**（对应 docs/04-industry-applications；商业项目管理应用）
- **转换关系**：**Business Objects** 应用**生命周期转换**（商业项目生命周期应用）；与 08-行业应用概念/03-商业项目管理、Category/01-Objects/04-Industry-Application-Objects 对应。

---

## 1. Overview / 概述

**English / 英文**:

Business objects represent business projects, business processes, and business outcomes in the category $\mathbf{Business}$. They capture business-specific project management patterns. This document provides a category-theoretic perspective on business objects, aligning with PMBOK 7th Edition and business management standards.

**中文**:

商业对象表示商业项目、商业过程和商业成果，属于范畴 $\mathbf{Business}$。它们捕捉商业特定的项目管理模式。本文档从范畴论视角提供商业对象的定义，对齐 PMBOK 第7版和商业管理标准。

**Key Insights / 关键洞察**:

- **Business Projects / 商业项目**: Business development projects / 商业开发项目
- **Business Processes / 商业过程**: Business operations / 商业运营
- **Business Outcomes / 商业成果**: Business value / 商业价值
- **Business Metrics / 商业指标**: ROI, revenue, profit / ROI、收入、利润

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Business / 商业范畴

**Definition 2.1** (Category $\mathbf{Business}$)

The category $\mathbf{Business}$ consists of:

- **Objects / 对象**: Business projects $P_{biz} \in \mathbf{Business}$
- **Morphisms / 态射**: Business transformations $f: P_1 \to P_2$
- **Composition / 复合**: Composition of transformations
- **Identity / 恒等**: Identity transformations

### 2.2 Business Object Properties / 商业对象性质

**Axiom 2.1** (Business Value)

Business objects have value:

$$\forall P_{biz}: Value(P_{biz}) \in \mathbb{R}^+$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Business Project Definition / 商业项目定义

**Definition 3.1** (Business Project)

A business project $P_{biz} \in \mathbf{Business}$:

$$P_{biz} = (Initiative, Timeline, BizTeam, Budget, ROI)$$

where:

- $Initiative$ - business initiative
- $Timeline$ - project timeline
- $BizTeam$ - business team
- $Budget$ - project budget
- $ROI$ - return on investment

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Business Value)

Business projects have value:

$$\forall P_{biz}: Value(P_{biz}) \geq 0$$

**Property 4.2** (Business ROI)

Business projects have ROI:

$$\forall P_{biz}: ROI(P_{biz}) \in \mathbb{R}$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Business → Project)

Business objects are projects:

$$Project: \mathbf{Business} \to \mathbf{Project}$$

**Relation 5.2** (Business → Resources)

Business objects use resources:

$$Resource: \mathbf{Business} \to \mathbf{Resource}$$

---

## 6. Examples / 例子

### 6.1 Product Launch Example / 产品发布例子

**Example 6.1** (Product Launch)

Consider product launch project:

$$P_{launch} = (NewProduct, 3Months, MarketingTeam, Budget, ExpectedROI)$$

with product launch components.

### 6.2 Market Expansion Example / 市场扩张例子

**Example 6.2** (Market Expansion)

Consider market expansion project:

$$P_{expansion} = (NewMarket, 6Months, SalesTeam, Budget, ExpectedROI)$$

with market expansion components.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Business Planning**: Planning business projects
- **ROI Management**: Managing return on investment
- **Value Delivery**: Delivering business value
- **Stakeholder Management**: Managing business stakeholders

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Business Project as Value Creation Engine / 商业项目即价值创造引擎)

商业对象 $P_{biz}=(Initiative, Timeline, BizTeam, Budget, ROI)$ 可看作一个**价值创造引擎**：$Initiative$ 是商业倡议（如产品发布、市场扩张），$Budget$ 是预算，$ROI$ 是投资回报率。范畴 $\mathbf{Business}$ 中的态射 $f: P_{launch} \to P_{expansion}$ 表示商业项目转换（如从产品发布扩展到市场扩张）。例如产品发布项目 $P_{launch}=(NewProduct, 3Months, MarketingTeam, Budget, ExpectedROI)$：$ExpectedROI=150\%$，$Budget=\$500K$；函子 $Resource: \mathbf{Business} \to \mathbf{Resource}$ 从商业项目中提取资源维度（团队、预算、时间），确保资源配置与 ROI 目标对齐。

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).

### 8.2 Related Files / 相关文件

- [Project Objects](01-Project-Objects.md)
- [Resource Objects](09-Resource-Objects.md)
- **docs**：`docs/04-industry-applications`（行业模型、商业项目管理；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
