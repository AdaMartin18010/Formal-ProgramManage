# Complexity and Systems Thinking Module / 复杂性与系统思维模块

## Overview / 概述

This module provides frameworks and theories for understanding and managing projects as complex systems. It bridges traditional project management with complexity science.

本模块提供理解和管理项目作为复杂系统的框架和理论。它将传统项目管理与复杂性科学联系起来。

## Contents / 内容

| Document | Description | 描述 |
|----------|-------------|------|
| [01-cynefin-framework.md](./01-cynefin-framework.md) | Cynefin decision framework | Cynefin决策框架 |
| [02-systems-dynamics.md](./02-systems-dynamics.md) | Systems dynamics modeling | 系统动力学建模 |
| [03-complex-adaptive-systems.md](./03-complex-adaptive-systems.md) | CAS theory for projects | 项目CAS理论 |

## Key Concepts / 关键概念

### Cynefin Framework

- Five domains: Clear, Complicated, Complex, Chaotic, Confused
- Context-appropriate decision making
- Domain-specific management approaches

### Systems Dynamics

- Stocks, flows, and feedback loops
- System archetypes in projects
- Counter-intuitive behavior modeling

### Complex Adaptive Systems

- Emergence and self-organization
- Simple rules and adaptation
- Edge of chaos optimization

## Relationship to PM Standards / 与PM标准关系

```mermaid
graph TD
    subgraph Complexity[Complexity Science]
        C1[Cynefin]
        C2[Systems Dynamics]
        C3[CAS Theory]
    end

    subgraph PMBOK[PMBOK 7th Edition]
        P1[Complexity Principle]
        P2[Systems Thinking Principle]
        P3[Adaptability Principle]
    end

    C1 --> P1
    C2 --> P2
    C3 --> P3
```

## When to Use / 何时使用

| Situation | Framework |
|-----------|-----------|
| Choosing management approach | Cynefin |
| Understanding project dynamics | Systems Dynamics |
| Designing team structures | CAS |
| Analyzing feedback loops | Systems Dynamics |
| Managing uncertainty | Cynefin + CAS |

## When to Use Formal Methods vs Cynefin / 何时用形式化方法 vs Cynefin

形式化方法（状态机、LTL/CTL、模型检验、定理证明）适合 **Clear** 与 **Complicated** 域：需求与因果关系可界定、状态与转换可穷举或可分析时，用 [形式化基础理论](../01-foundations/README.md) 与 [形式化验证](../03-formal-verification/verification-theory.md) 可得到严格的可验证结论。
**Complex** 域（涌现、不可预测）宜用 Cynefin 的 **Probe–Sense–Respond**：先小规模试验再感知、再响应，而非先建完整形式规范。**Chaotic** 域需先稳定再归类，形式化可后续引入。
详见 [Cynefin 框架](./01-cynefin-framework.md) 与 [形式化验证理论](../03-formal-verification/verification-theory.md) 的交叉引用。

## Authority Sources / 权威来源

- Dave Snowden (Cynefin)
- Jay Forrester, John Sterman (Systems Dynamics)
- Santa Fe Institute, John Holland (CAS)
- Ralph Stacey (Complexity and Management)

**MIT ESD.36 对标**：本模块与 [docs/README.md](../README.md) 大学课程对标表中的 **MIT ESD.36 — System Project Management** 对应。系统动力学见 [02-systems-dynamics.md](./02-systems-dynamics.md)；设计结构矩阵 (DSM)、CPM、PERT 与生命周期/进度的对应见 [02-project-management/lifecycle-models.md](../02-project-management/lifecycle-models.md) §2.1.6 设计结构矩阵 (DSM) 与 MIT ESD.36 对标。

---

**Last Updated / 最后更新**: 2026-02-02
**Version / 版本**: 1.0
**Status / 状态**: ✅ Complete
