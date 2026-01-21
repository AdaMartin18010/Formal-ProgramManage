# 2. 项目管理核心模型

## 2.1 概述

项目管理核心模型是Formal-ProgramManage的核心理论体系，定义了项目管理的四个核心维度：生命周期、资源、风险和质量。本理论体系严格对标PMBOK 7th Edition、ISO 21500:2012、ISO 31000:2018、ISO/IEC 25010、PRINCE2 2017、CMMI-DEV等国际项目管理标准。

### 🎯 核心特色

- **标准对标**: 严格对标PMBOK、ISO、PRINCE2等国际标准
- **形式化规范**: 基于严格的数学定义和形式化模型
- **算法实现**: 提供完整的Rust代码实现
- **实践导向**: 结合理论模型与实际应用场景
- **系统集成**: 四个核心模型相互关联，形成完整体系

## 2.2 目录结构

### 2.2.1 生命周期模型

- **[2.1 项目生命周期模型](./lifecycle-models.md)** - 项目从启动到收尾的完整演进过程
  - 对标标准：PMBOK 7th Edition、ISO 21500:2012、PRINCE2 2017、APM Body of Knowledge 7th Edition
  - 核心内容：生命周期基础理论、标准生命周期模型、生命周期优化、生命周期验证

### 2.2.2 资源管理模型

- **[2.2 资源管理模型](./resource-models.md)** - 项目资源的优化配置、分配和监控机制
  - 对标标准：PMBOK 7th Edition、ISO 21500、PRINCE2
  - 核心内容：资源管理基础理论、资源优化模型、资源分配算法、资源监控系统

### 2.2.3 风险管理模型

- **[2.3 风险管理模型](./risk-models.md)** - 项目风险的识别、分析、应对和监控机制
  - 对标标准：PMBOK 7th Edition、ISO 31000、PRINCE2
  - 核心内容：风险管理基础理论、风险识别模型、风险分析模型、风险应对模型、风险监控模型

### 2.2.4 质量管理模型

- **[2.4 质量管理模型](./quality-models.md)** - 项目质量的规划、保证、控制和改进机制
  - 对标标准：ISO/IEC 25010、ISO 9001、CMMI-DEV
  - 核心内容：质量管理基础理论、质量规划模型、质量保证模型、质量控制模型、质量改进模型

## 2.3 形式化规范

### 2.3.1 项目管理模型定义

**定义 2.0.1** (项目管理核心模型) 项目管理核心模型是一个四元组：
$$\mathcal{PM} = (\mathcal{L}, \mathcal{R}_{res}, \mathcal{R}_{risk}, \mathcal{Q})$$

其中：

- $\mathcal{L}$ 是生命周期模型，满足 $\mathcal{L} = (P, T, G, C)$
- $\mathcal{R}_{res}$ 是资源管理模型，满足 $\mathcal{R}_{res} = (H, M, T, F)$
- $\mathcal{R}_{risk}$ 是风险管理模型，满足 $\mathcal{R}_{risk} = (E, P, I, T, C)$
- $\mathcal{Q}$ 是质量管理模型，满足 $\mathcal{Q} = (F, E, M, P, S, U)$

### 2.3.2 模型一致性公理

**公理 2.0.1** (生命周期-资源一致性) 对于任意项目阶段 $p \in P$：
$$\sum_{r \in \mathcal{R}_{res}} \text{allocate}(p, r) \leq \text{available}(r)$$

**公理 2.0.2** (风险-质量一致性) 对于任意风险事件 $e \in E$：
$$\text{Impact}(e) \leq 1 - \text{Quality}(\text{affected\_component})$$

**公理 2.0.3** (资源-风险一致性) 对于任意资源 $r \in \mathcal{R}_{res}$：
$$\text{Risk}(r) \propto \frac{\text{utilization}(r)}{\text{capacity}(r)}$$

### 2.3.3 模型集成函数

**定义 2.0.2** (模型集成) 模型集成函数：
$$\text{Integrate}: \mathcal{L} \times \mathcal{R}_{res} \times \mathcal{R}_{risk} \times \mathcal{Q} \rightarrow \mathcal{PM}$$

定义为：
$$\text{Integrate}(\mathcal{L}, \mathcal{R}_{res}, \mathcal{R}_{risk}, \mathcal{Q}) = \mathcal{PM}$$

满足：
- $\forall p \in P: \text{resources}(p) \subseteq \mathcal{R}_{res}$
- $\forall p \in P: \text{risks}(p) \subseteq \mathcal{R}_{risk}$
- $\forall p \in P: \text{quality}(p) \in \mathcal{Q}$

## 2.4 思维导图

```mermaid
graph TB
    A[项目管理核心模型] --> B[2.1 生命周期模型]
    A --> C[2.2 资源管理模型]
    A --> D[2.3 风险管理模型]
    A --> E[2.4 质量管理模型]

    B --> B1[启动阶段]
    B --> B2[规划阶段]
    B --> B3[执行阶段]
    B --> B4[监控阶段]
    B --> B5[收尾阶段]

    C --> C1[人力资源]
    C --> C2[物质资源]
    C --> C3[技术资源]
    C --> C4[财务资源]

    D --> D1[风险识别]
    D --> D2[风险分析]
    D --> D3[风险应对]
    D --> D4[风险监控]

    E --> E1[质量规划]
    E --> E2[质量保证]
    E --> E3[质量控制]
    E --> E4[质量改进]

    B -.-> C
    B -.-> D
    B -.-> E
    C -.-> D
    C -.-> E
    D -.-> E
```

## 2.5 模型关系矩阵

| 模型 | 生命周期 | 资源管理 | 风险管理 | 质量管理 |
|------|---------|---------|---------|---------|
| **生命周期** | - | 资源分配 | 风险触发 | 质量目标 |
| **资源管理** | 阶段资源需求 | - | 资源风险 | 资源质量 |
| **风险管理** | 阶段风险 | 资源约束 | - | 质量风险 |
| **质量管理** | 阶段质量 | 资源质量 | 风险影响 | - |

## 2.6 实现要求

### 2.6.1 代码规范

所有实现必须包含：

- 形式化定义的结构体
- 核心算法实现
- 验证函数
- 测试用例
- 文档注释

### 2.6.2 验证要求

每个模型必须通过：

- 模型一致性检查
- 算法正确性验证
- 性能测试
- 集成测试

### 2.6.3 标准对标

每个模型必须明确标注：

- 对标的国际标准
- 标准版本号
- 标准对应章节
- 实现差异说明

## 2.7 引用关系

### 2.7.1 内部引用

- 生命周期模型 ↔ 资源管理模型：资源分配与阶段规划
- 生命周期模型 ↔ 风险管理模型：风险触发与阶段转换
- 生命周期模型 ↔ 质量管理模型：质量目标与阶段交付
- 资源管理模型 ↔ 风险管理模型：资源约束与风险应对
- 资源管理模型 ↔ 质量管理模型：资源质量与质量保证
- 风险管理模型 ↔ 质量管理模型：风险影响与质量改进

### 2.7.2 外部引用

- **基础理论**：参见 [1.1 形式化基础理论](../01-foundations/README.md)
- **数学模型**：参见 [1.2 数学模型基础](../01-foundations/mathematical-models.md)
- **语义模型**：参见 [1.3 语义模型理论](../01-foundations/semantic-models.md)
- **形式化验证**：参见 [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)
- **模型检验**：参见 [3.2 模型检验方法](../03-formal-verification/model-checking.md)
- **定理证明**：参见 [3.3 定理证明系统](../03-formal-verification/theorem-proving.md)

## 2.8 国际标准对标

### 2.8.1 PMBOK 7th Edition

- **知识领域**: 10个知识领域
- **过程组**: 5个过程组
- **绩效域**: 8个绩效域
- **价值交付**: 价值交付系统

### 2.8.2 ISO 标准

- **ISO 21500:2012**: 项目管理指南
- **ISO 31000:2018**: 风险管理指南
- **ISO/IEC 25010:2011**: 软件质量模型
- **ISO 9001:2015**: 质量管理体系

### 2.8.3 PRINCE2 2017

- **主题**: 7个主题
- **过程**: 7个过程
- **原则**: 7个原则

### 2.8.4 CMMI-DEV

- **过程域**: 22个过程域
- **成熟度等级**: 5个成熟度等级

## 2.9 参考文献

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
3. ISO 31000:2018. Risk management - Guidelines. International Organization for Standardization.
4. ISO/IEC 25010:2011. Systems and software Quality Requirements and Evaluation (SQuaRE) - System and software quality models.
5. ISO 9001:2015. Quality management systems - Requirements.
6. AXELOS. (2017). Managing Successful Projects with PRINCE2 2017 Edition. TSO (The Stationery Office).
7. CMMI Product Team. (2010). CMMI for Development, Version 1.3. Software Engineering Institute.
8. Association for Project Management. (2019). APM Body of Knowledge 7th Edition. APM.
9. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling (12th ed.). John Wiley & Sons.
10. Meredith, J. R., & Mantel, S. J. (2019). Project management: a managerial approach (10th ed.). John Wiley & Sons.

---

**最后更新**: 2025-01-XX
**维护者**: Formal-ProgramManage团队
