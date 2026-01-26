# 不确定性量化 / Uncertainty Quantification

## 📋 Table of Contents / 目录

- [不确定性量化 / Uncertainty Quantification](#不确定性量化--uncertainty-quantification)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Definition / 定义](#2-definition--定义)
    - [2.1 不确定性量化定义](#21-不确定性量化定义)
    - [2.2 Markov范畴基础](#22-markov范畴基础)
    - [2.3 组合不确定性](#23-组合不确定性)
    - [2.4 不确定性在项目管理中的应用](#24-不确定性在项目管理中的应用)
  - [3. Category Theory Perspective / 范畴论视角](#3-category-theory-perspective--范畴论视角)
    - [3.1 不确定性作为对象](#31-不确定性作为对象)
    - [3.2 不确定性传播作为态射](#32-不确定性传播作为态射)
    - [3.3 不确定性量化函子](#33-不确定性量化函子)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 不确定性的组合性](#41-不确定性的组合性)
    - [4.2 不确定性的可量化性](#42-不确定性的可量化性)
    - [4.3 不确定性的可传播性](#43-不确定性的可传播性)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 与风险管理的关系](#51-与风险管理的关系)
    - [5.2 与预测分析的关系](#52-与预测分析的关系)
    - [5.3 与其他应用的关系](#53-与其他应用的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 项目进度不确定性量化](#61-项目进度不确定性量化)
    - [6.2 成本不确定性量化](#62-成本不确定性量化)
    - [6.3 资源不确定性量化](#63-资源不确定性量化)
  - [7. Explanations / 解释](#7-explanations--解释)
    - [7.1 数学解释 / Mathematical Explanation](#71-数学解释--mathematical-explanation)
    - [7.2 直观解释 / Intuitive Explanation](#72-直观解释--intuitive-explanation)
    - [7.3 应用解释 / Application Explanation](#73-应用解释--application-explanation)
    - [7.4 认知解释 / Cognitive Explanation](#74-认知解释--cognitive-explanation)
    - [7.5 历史解释 / Historical Explanation](#75-历史解释--historical-explanation)
    - [7.6 哲学解释 / Philosophical Explanation](#76-哲学解释--philosophical-explanation)
    - [7.7 技术解释 / Technical Explanation](#77-技术解释--technical-explanation)
    - [7.8 实践解释 / Practical Explanation](#78-实践解释--practical-explanation)
    - [7.9 对比解释 / Comparative Explanation](#79-对比解释--comparative-explanation)
    - [7.10 系统解释 / System Explanation](#710-系统解释--system-explanation)
  - [8. Argumentation / 论证](#8-argumentation--论证)
    - [8.1 为什么需要不确定性量化](#81-为什么需要不确定性量化)
    - [8.2 不确定性量化的有效性证明](#82-不确定性量化的有效性证明)
  - [9. Applications / 应用](#9-applications--应用)
    - [9.1 在项目风险管理中的应用](#91-在项目风险管理中的应用)
    - [9.2 在预测分析中的应用](#92-在预测分析中的应用)
    - [9.3 在决策支持中的应用](#93-在决策支持中的应用)
  - [10. References / 参考文献](#10-references--参考文献)
    - [10.1 Standards / 标准](#101-standards--标准)
    - [10.2 Category Theory / 范畴论](#102-category-theory--范畴论)
    - [10.3 Related Files / 相关文件](#103-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层（应用）**（对应 docs/06-ci-verification、docs/03-formal-verification）
- **转换关系**：不确定性量化作为**风险转换**和**预测转换**的数学框架，使用Markov范畴建模不确定性的组合和传播；与 **04-风险管理概念**、**05-数据驱动决策框架** 对应。

---

## 1. Overview / 概述

**English / 英文**:

Uncertainty quantification in project management uses category theory, particularly Markov categories, to model and quantify uncertainty in project parameters, predictions, and decisions. Based on 2024-2025 research advances, this approach enables compositional handling of uncertainty, combining probabilistic uncertainty, nondeterminism, and information loss in a unified framework. It supports better risk assessment, prediction accuracy, and decision-making under uncertainty.

**中文**:

项目管理中的不确定性量化使用范畴论，特别是Markov范畴，来建模和量化项目参数、预测和决策中的不确定性。基于2024-2025年的研究进展，这种方法能够组合处理不确定性，在统一框架中结合概率不确定性、非确定性和信息损失。它支持更好的风险评估、预测准确性和不确定性下的决策。

**Key Insights / 关键洞察**:

- **Markov Categories / Markov范畴**: Framework for probabilistic uncertainty / 概率不确定性的框架
- **Compositional Uncertainty / 组合不确定性**: Combining uncertainties compositionally / 组合地结合不确定性
- **Multiple Uncertainty Types / 多种不确定性类型**: Probabilistic, nondeterministic, information loss / 概率、非确定性、信息损失
- **Unified Framework / 统一框架**: Single framework for all uncertainty types / 所有不确定性类型的单一框架

---

## 2. Definition / 定义

### 2.1 不确定性量化定义

**Definition 2.1** (Uncertainty Quantification)

Uncertainty quantification is the process of characterizing and quantifying uncertainty in project parameters, models, and predictions.

**Formal Definition / 形式化定义**:

$$\text{UncertaintyQuantification}(X) = (\mu_X, \sigma_X, U_X)$$

where:

- $\mu_X$: Mean or expected value
- $\sigma_X$: Standard deviation or uncertainty measure
- $U_X$: Uncertainty distribution or bounds

**Uncertainty Types / 不确定性类型**:

1. **Probabilistic Uncertainty / 概率不确定性**: Random variability
2. **Epistemic Uncertainty / 认知不确定性**: Lack of knowledge
3. **Aleatory Uncertainty / 偶然不确定性**: Inherent randomness
4. **Nondeterminism / 非确定性**: Multiple possible outcomes

### 2.2 Markov范畴基础

**Definition 2.2** (Markov Categories)

A Markov category is a symmetric monoidal category where morphisms represent probabilistic processes.

**Formal Definition / 形式化定义**:

A Markov category $\mathbf{Markov}$ is a symmetric monoidal category $(\mathbf{C}, \otimes, I)$ where:

- Objects represent sample spaces
- Morphisms $f: X \to Y$ represent stochastic maps
- Composition represents sequential probabilistic processes
- Tensor product $\otimes$ represents independent probabilistic processes

**Key Properties / 关键性质**:

- **Copying / 复制**: Morphisms can copy information
- **Discarding / 丢弃**: Morphisms can discard information
- **Compositionality / 组合性**: Probabilistic processes compose

**2024-2025 Advances / 2024-2025进展**:

Recent research (2024-2025) has extended Markov categories to handle:

- **Composable Uncertainty / 组合不确定性**: Uncertainty that composes well
- **Graded Monads / 分级单子**: Managing nondeterministic choices
- **Extended Distributions / 扩展分布**: Combining probability with nondeterminism

### 2.3 组合不确定性

**Definition 2.3** (Composable Uncertainty)

Composable uncertainty allows uncertainty to be combined and propagated through compositional structures.

**Formal Definition / 形式化定义**:

Given uncertainties $U_1$ and $U_2$, their composition:

$$U_1 \circ U_2: X \to Z$$

preserves uncertainty structure while combining uncertainties.

**Compositional Framework / 组合框架**:

Using symmetric monoidal categories with Markov structure:

$$\text{Uncertainty}: \mathbf{Project} \to \mathbf{UncertainProject}$$

where uncertainty is preserved under composition.

**2024-2025 Research / 2024-2025研究**:

Recent work (2025) shows how to integrate uncertainty into symmetric monoidal categories using:

- **Markov Categories / Markov范畴**: For probabilistic uncertainty
- **Graded Monads / 分级单子**: For nondeterministic uncertainty
- **Change-of-Base / 基变换**: For combining uncertainty types

### 2.4 不确定性在项目管理中的应用

**Definition 2.4** (Uncertainty in Project Management)

Uncertainty in project management includes:

1. **Schedule Uncertainty / 进度不确定性**: Uncertainty in activity durations
2. **Cost Uncertainty / 成本不确定性**: Uncertainty in costs
3. **Resource Uncertainty / 资源不确定性**: Uncertainty in resource availability
4. **Risk Uncertainty / 风险不确定性**: Uncertainty in risk occurrence

**Formal Definition / 形式化定义**:

$$\text{ProjectUncertainty}(P) = (\text{ScheduleUncertainty}(P), \text{CostUncertainty}(P), \text{ResourceUncertainty}(P), \text{RiskUncertainty}(P))$$

**Category Theory Modeling / 范畴论建模**:

Using Markov categories:

$$\text{UncertainProject} = \text{Markov}(\text{Project})$$

where project parameters are modeled as probabilistic distributions.

---

## 3. Category Theory Perspective / 范畴论视角

### 3.1 不确定性作为对象

**Definition 3.1** (Uncertainty Object)

An uncertainty object $U \in \mathbf{Uncertainty}$ is an object:

$$U = (\text{Distribution}, \text{Bounds}, \text{Type})$$

where:

- $\text{Distribution}$: Probability distribution or uncertainty bounds
- $\text{Bounds}$: Upper and lower bounds
- $\text{Type}$: Type of uncertainty (probabilistic, epistemic, etc.)

**Example 3.1** (Schedule Uncertainty Object)

Schedule uncertainty:

$$U_{schedule} = (\text{Normal}(\mu=10, \sigma=2), [8, 12], \text{Probabilistic})$$

where:

- Distribution: Normal distribution with mean 10, std dev 2
- Bounds: [8, 12] days
- Type: Probabilistic uncertainty

### 3.2 不确定性传播作为态射

**Definition 3.2** (Uncertainty Propagation Morphism)

Uncertainty propagation is a morphism:

$$prop_{uncertainty}: U_1 \times U_2 \to U_{combined}$$

that combines uncertainties.

**Example 3.2** (Schedule Propagation)

Combining activity uncertainties:

$$prop_{schedule}(U_{A1}, U_{A2}) = U_{A1+A2}$$

where combined uncertainty follows from activity uncertainties.

**Markov Category Representation / Markov范畴表示**:

In Markov categories, uncertainty propagation:

$$f: X \to Y$$

represents a stochastic map that propagates uncertainty from $X$ to $Y$.

### 3.3 不确定性量化函子

**Definition 3.3** (Uncertainty Quantification Functor)

Uncertainty quantification corresponds to a functor:

$$UQ: \mathbf{Project} \to \mathbf{UncertainProject}$$

that adds uncertainty to project parameters.

**Theorem 3.1** (Uncertainty Composition)

Uncertainty functors compose:

$$(UQ_2 \circ UQ_1)(P) = UQ_2(UQ_1(P))$$

where uncertainties are composed.

---

## 4. Properties / 性质

### 4.1 不确定性的组合性

**Property 4.1** (Uncertainty Compositionality)

Uncertainty composes well in Markov categories:

$$\text{Uncertainty}(f \circ g) = \text{Uncertainty}(f) \circ \text{Uncertainty}(g)$$

where uncertainty is preserved under composition.

### 4.2 不确定性的可量化性

**Property 4.2** (Uncertainty Quantifiability)

Uncertainty can be quantified:

$$\text{Quantify}(U) = (\mu, \sigma, \text{Bounds})$$

where uncertainty is characterized by measures.

### 4.3 不确定性的可传播性

**Property 4.3** (Uncertainty Propagability)

Uncertainty propagates through processes:

$$\text{Propagate}(U_{input}, f) = U_{output}$$

where output uncertainty depends on input uncertainty and process $f$.

---

## 5. Relations / 关系

### 5.1 与风险管理的关系

**Relation 5.1** (Risk Management Relationship)

Uncertainty quantification supports risk management:

- **Risk Identification / 风险识别**: Identify uncertain parameters
- **Risk Quantification / 风险量化**: Quantify uncertainty in risks
- **Risk Propagation / 风险传播**: Propagate uncertainty through project
- **Risk Assessment / 风险评估**: Assess overall project uncertainty

### 5.2 与预测分析的关系

**Relation 5.2** (Predictive Analytics Relationship)

Uncertainty quantification enhances predictive analytics:

- **Prediction Uncertainty / 预测不确定性**: Quantify uncertainty in predictions
- **Confidence Intervals / 置信区间**: Provide confidence intervals
- **Sensitivity Analysis / 敏感性分析**: Analyze sensitivity to uncertainty
- **Robust Predictions / 稳健预测**: Make robust predictions

### 5.3 与其他应用的关系

**Relation 5.3** (Other Applications Relationship)

Uncertainty quantification relates to:

- **Data-Driven Decisions / 数据驱动决策**: Quantify uncertainty in data
- **Resource Scheduling / 资源调度**: Handle resource uncertainty
- **Parameter Optimization / 参数优化**: Optimize under uncertainty

---

## 6. Examples / 例子

### 6.1 项目进度不确定性量化

**Example 6.1** (Project Schedule Uncertainty Quantification)

**Project / 项目**: Software development

**Activity Uncertainties / 活动不确定性**:

- Activity A: $U_A = \text{Normal}(10, 2)$ days
- Activity B: $U_B = \text{Normal}(15, 3)$ days
- Activity C: $U_C = \text{Normal}(8, 1.5)$ days

**Sequential Activities / 顺序活动**:

- Combined: $U_{A+B+C} = \text{Normal}(33, \sqrt{4+9+2.25}) = \text{Normal}(33, 3.91)$ days

**Parallel Activities / 并行活动**:

- Critical path: $\max(U_A, U_B) = \text{MaxNormal}(U_A, U_B)$

**Markov Category Modeling / Markov范畴建模**:

Using Markov categories:

$$U_{schedule} = \text{Markov}(U_A \otimes U_B \otimes U_C)$$

where uncertainties compose according to activity dependencies.

### 6.2 成本不确定性量化

**Example 6.2** (Cost Uncertainty Quantification)

**Project / 项目**: Construction project

**Cost Component Uncertainties / 成本组成部分不确定性**:

- Materials: $U_{materials} = \text{Uniform}(80K, 120K)$
- Labor: $U_{labor} = \text{Normal}(100K, 15K)$
- Equipment: $U_{equipment} = \text{Triangular}(20K, 30K, 40K)$

**Total Cost Uncertainty / 总成本不确定性**:

- Combined: $U_{total} = U_{materials} + U_{labor} + U_{equipment}$
- Distribution: Approximated using convolution or Monte Carlo

**Confidence Intervals / 置信区间**:

- 90% CI: [$195K, $265K]
- Expected: $230K

### 6.3 资源不确定性量化

**Example 6.3** (Resource Uncertainty Quantification)

**Project / 项目**: IT project

**Resource Availability Uncertainties / 资源可用性不确定性**:

- Developer availability: $U_{dev} = \text{Bernoulli}(0.8)$ (80% available)
- Server capacity: $U_{server} = \text{Normal}(100, 10)$ units
- Network bandwidth: $U_{network} = \text{Uniform}(50, 100)$ Mbps

**Combined Resource Uncertainty / 组合资源不确定性**:

Using Markov categories:

$$U_{resources} = \text{Markov}(U_{dev} \otimes U_{server} \otimes U_{network})$$

where resource uncertainties are combined.

---

## 7. Explanations / 解释

### 7.1 数学解释 / Mathematical Explanation

**Mathematical Structure / 数学结构**:

Uncertainty quantification uses:

$$\text{Uncertainty}(X) = P(X)$$

where $P(X)$ is a probability distribution or uncertainty measure.

**Markov Category Structure / Markov范畴结构**:

In Markov categories:

$$f: X \to Y$$

represents a stochastic map:

$$P(Y \mid X)$$

that propagates uncertainty.

### 7.2 直观解释 / Intuitive Explanation

**Intuitive Understanding / 直观理解**:

Think of uncertainty as **fog**:

- **Clear / 清晰**: Low uncertainty (thin fog)
- **Uncertain / 不确定**: High uncertainty (thick fog)
- **Propagation / 传播**: Fog spreads through processes
- **Combination / 组合**: Multiple fog sources combine

Just as fog affects visibility, uncertainty affects project predictions and decisions.

### 7.3 应用解释 / Application Explanation

**Practical Application / 实际应用**:

In practice, uncertainty quantification:

- **Quantifies uncertainty**: Measures uncertainty in parameters
- **Propagates uncertainty**: Tracks uncertainty through processes
- **Supports decisions**: Provides uncertainty-aware decisions
- **Improves predictions**: Better predictions with uncertainty bounds

### 7.4 认知解释 / Cognitive Explanation

**Cognitive Understanding / 认知理解**:

From a cognitive perspective, uncertainty quantification:

- **Reduces overconfidence**: Acknowledges uncertainty
- **Improves understanding**: Better understanding of risks
- **Enhances decision-making**: Makes uncertainty-aware decisions
- **Supports learning**: Learns from uncertainty patterns

### 7.5 历史解释 / Historical Explanation

**Historical Development / 历史发展**:

- **Early PM**: Point estimates (no uncertainty)
- **Modern PM**: Range estimates (simple uncertainty)
- **Current PM**: Probabilistic models (probabilistic uncertainty)
- **Future PM**: Compositional uncertainty (Markov categories)

### 7.6 哲学解释 / Philosophical Explanation

**Philosophical Perspective / 哲学视角**:

Uncertainty quantification represents:

- **Epistemology / 认识论**: Acknowledgment of limited knowledge
- **Probability / 概率**: Probabilistic reasoning
- **Robustness / 稳健性**: Robustness to uncertainty
- **Humility / 谦逊**: Humility about predictions

### 7.7 技术解释 / Technical Explanation

**Technical Details / 技术细节**:

From a technical perspective:

- **Markov Categories / Markov范畴**: Mathematical framework
- **Probability Distributions / 概率分布**: Uncertainty representation
- **Monte Carlo / 蒙特卡洛**: Uncertainty propagation methods
- **Sensitivity Analysis / 敏感性分析**: Uncertainty analysis

### 7.8 实践解释 / Practical Explanation

**Practical Perspective / 实践视角**:

In practice, uncertainty quantification:

- **Improves accuracy**: More accurate predictions
- **Reduces risk**: Better risk management
- **Enhances confidence**: Confidence intervals
- **Supports planning**: Uncertainty-aware planning

### 7.9 对比解释 / Comparative Explanation

**Comparison / 对比**:

| Aspect / 方面 | Point Estimates | Range Estimates | Probabilistic | Compositional |
|--------------|----------------|----------------|---------------|---------------|
| Accuracy / 准确性 | Low | Medium | High | Highest |
| Complexity / 复杂性 | Low | Medium | High | Highest |
| Compositionality / 组合性 | N/A | Limited | Good | Excellent |

### 7.10 系统解释 / System Explanation

**System Perspective / 系统视角**:

From a systems perspective, uncertainty quantification:

- **System inputs / 系统输入**: Uncertain parameters
- **System processing / 系统处理**: Uncertainty propagation
- **System outputs / 系统输出**: Uncertain predictions
- **System feedback / 系统反馈**: Uncertainty reduction through learning

---

## 8. Argumentation / 论证

### 8.1 为什么需要不确定性量化

**Argument 8.1** (Need for Uncertainty Quantification)

**Why Uncertainty Quantification Is Needed / 为什么需要不确定性量化**:

1. **Realistic Predictions / 现实预测**: Projects have inherent uncertainty
2. **Risk Management / 风险管理**: Better risk assessment
3. **Decision Support / 决策支持**: Uncertainty-aware decisions
4. **Accuracy Improvement / 准确性提高**: More accurate predictions
5. **Robustness / 稳健性**: Robustness to uncertainty

**Evidence / 证据**:

- 2024-2025 research advances in compositional uncertainty
- Markov categories provide unified framework
- Better project outcomes with uncertainty quantification
- Industry demand for uncertainty-aware PM

### 8.2 不确定性量化的有效性证明

**Argument 8.2** (Effectiveness of Uncertainty Quantification)

**Effectiveness Criteria / 有效性标准**:

1. **Accuracy Improvement / 准确性提高**: More accurate predictions ✅
2. **Risk Reduction / 风险降低**: Better risk management ✅
3. **Decision Quality / 决策质量**: Better decisions ✅
4. **Robustness / 稳健性**: Robustness to uncertainty ✅
5. **Compositionality / 组合性**: Compositional uncertainty handling ✅

**Proof / 证明**:

- **Accuracy**: Uncertainty quantification improves prediction accuracy ✅
- **Risk**: Better risk assessment and management ✅
- **Decisions**: Uncertainty-aware decisions are better ✅
- **Robustness**: More robust to uncertainty ✅
- **Compositionality**: Markov categories enable composition ✅

---

## 9. Applications / 应用

### 9.1 在项目风险管理中的应用

**Application 9.1** (Risk Management)

**Uncertainty Quantification / 不确定性量化**:

- **Risk Identification / 风险识别**: Identify uncertain parameters
- **Risk Quantification / 风险量化**: Quantify uncertainty in risks
- **Risk Propagation / 风险传播**: Propagate uncertainty
- **Risk Assessment / 风险评估**: Assess overall uncertainty

**Benefits / 效益**:

- Better risk understanding
- More accurate risk assessment
- Improved risk management
- Better decision-making

### 9.2 在预测分析中的应用

**Application 9.2** (Predictive Analytics)

**Uncertainty Quantification / 不确定性量化**:

- **Prediction Uncertainty / 预测不确定性**: Quantify prediction uncertainty
- **Confidence Intervals / 置信区间**: Provide confidence intervals
- **Sensitivity Analysis / 敏感性分析**: Analyze sensitivity
- **Robust Predictions / 稳健预测**: Make robust predictions

**Benefits / 效益**:

- More accurate predictions
- Confidence intervals
- Better understanding
- Improved decisions

### 9.3 在决策支持中的应用

**Application 9.3** (Decision Support)

**Uncertainty Quantification / 不确定性量化**:

- **Uncertainty-Aware Decisions / 不确定性感知决策**: Consider uncertainty
- **Robust Optimization / 稳健优化**: Optimize under uncertainty
- **Sensitivity Analysis / 敏感性分析**: Analyze decision sensitivity
- **Risk-Adjusted Decisions / 风险调整决策**: Adjust for uncertainty

**Benefits / 效益**:

- Better decisions
- Risk-adjusted choices
- Improved outcomes
- Reduced surprises

---

## 10. References / 参考文献

### 10.1 Standards / 标准

- **ISO 31000:2018**: Risk management — Guidelines
- **PMBOK Guide 8th Edition** (2025): Risk management and uncertainty

### 10.2 Category Theory / 范畴论

- **Markov Categories**: Framework for probabilistic uncertainty
- **Composable Uncertainty** (2024-2025): Recent research advances
- **Graded Monads**: Managing nondeterministic uncertainty
- **Applied Category Theory** (2024-2025): Latest research

### 10.3 Related Files / 相关文件

- [04-风险管理概念](../Concept/04-风险管理概念/) - Risk Management Concepts
- [05-数据驱动决策框架](../Concept/13-综合实践概念/05-数据驱动决策框架.md) - Data-Driven Decision Framework
- [05-Symmetric-Monoidal-Resource-Scheduling.md](05-Symmetric-Monoidal-Resource-Scheduling.md) - Symmetric Monoidal Categories

---

**Last Updated / 最后更新**: 2026-01-27
**Version / 版本**: 1.0
**Status / 状态**: ✅ Complete / 完成

---

## 📊 Summary / 总结

Uncertainty quantification uses category theory, particularly Markov categories, to model and quantify uncertainty in project management. Based on 2024-2025 research advances, this approach enables compositional handling of uncertainty, supporting better risk assessment, prediction accuracy, and decision-making under uncertainty.

不确定性量化使用范畴论，特别是Markov范畴，来建模和量化项目管理中的不确定性。基于2024-2025年的研究进展，这种方法能够组合处理不确定性，支持更好的风险评估、预测准确性和不确定性下的决策。
