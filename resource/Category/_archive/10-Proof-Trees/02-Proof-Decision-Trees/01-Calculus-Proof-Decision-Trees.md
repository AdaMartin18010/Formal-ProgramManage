# Calculus Proof Decision Trees / 微积分证明决策树

## 📋 Table of Contents / 目录

- [Calculus Proof Decision Trees / 微积分证明决策树](#calculus-proof-decision-trees--微积分证明决策树)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Limit Proof Decision Trees / 极限证明决策树](#2-limit-proof-decision-trees--极限证明决策树)
    - [2.1 Limit Existence Decision Tree / 极限存在性决策树](#21-limit-existence-decision-tree--极限存在性决策树)
    - [2.2 Limit Uniqueness Decision Tree / 极限唯一性决策树](#22-limit-uniqueness-decision-tree--极限唯一性决策树)
  - [3. Derivative Proof Decision Trees / 导数证明决策树](#3-derivative-proof-decision-trees--导数证明决策树)
    - [3.1 Differentiability Proof Decision Tree / 可微性证明决策树](#31-differentiability-proof-decision-tree--可微性证明决策树)
    - [3.2 Chain Rule Proof Decision Tree / 链式法则证明决策树](#32-chain-rule-proof-decision-tree--链式法则证明决策树)
  - [4. Integral Proof Decision Trees / 积分证明决策树](#4-integral-proof-decision-trees--积分证明决策树)
    - [4.1 Integrability Proof Decision Tree / 可积性证明决策树](#41-integrability-proof-decision-tree--可积性证明决策树)
    - [4.2 Fundamental Theorem Proof Decision Tree / 微积分基本定理证明决策树](#42-fundamental-theorem-proof-decision-tree--微积分基本定理证明决策树)
  - [5. Series Proof Decision Trees / 级数证明决策树](#5-series-proof-decision-trees--级数证明决策树)
    - [5.1 Convergence Proof Decision Tree / 收敛性证明决策树](#51-convergence-proof-decision-tree--收敛性证明决策树)
  - [6. Proof Strategy Selection / 证明策略选择](#6-proof-strategy-selection--证明策略选择)
    - [6.1 When to Use Direct Construction / 何时使用直接构造](#61-when-to-use-direct-construction--何时使用直接构造)
    - [6.2 When to Use ε-δ Definition / 何时使用ε-δ定义](#62-when-to-use-ε-δ-definition--何时使用ε-δ定义)
    - [6.3 When to Use Mean Value Theorem / 何时使用中值定理](#63-when-to-use-mean-value-theorem--何时使用中值定理)
    - [6.4 When to Use Category Theory / 何时使用范畴论](#64-when-to-use-category-theory--何时使用范畴论)
  - [7. Detailed Proof Examples / 详细证明示例](#7-detailed-proof-examples--详细证明示例)
    - [Example 1: Limit Existence Proof / 例子1：极限存在性证明](#example-1-limit-existence-proof--例子1极限存在性证明)
    - [Example 2: Chain Rule Proof / 例子2：链式法则证明](#example-2-chain-rule-proof--例子2链式法则证明)
    - [Example 3: Fundamental Theorem Proof / 例子3：微积分基本定理证明](#example-3-fundamental-theorem-proof--例子3微积分基本定理证明)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Mathematical References / 数学参考文献](#81-mathematical-references--数学参考文献)
    - [8.2 International Standards / 国际标准](#82-international-standards--国际标准)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)
    - [**2026-2027框架对齐说明 / 2026-2027 Framework Alignment**](#2026-2027框架对齐说明--2026-2027-framework-alignment)

---

## 1. Overview / 概述

**English / 英文**:

This document provides proof decision trees for calculus concepts (limits, derivatives, integrals, series), guiding proof strategy selection and step-by-step proof construction. **Updated for 2026-2027**: Enhanced with cognitive-friendly representations and multiple proof perspectives.

**中文**:

本文档提供微积分概念（极限、导数、积分、级数）的证明决策树，指导证明策略选择和分步证明构造。**2026-2027更新**：增强认知友好型表征和多种证明视角。

**Key Insights / 关键洞察**:

- **Proof Strategy / 证明策略**: Different proof techniques for different calculus concepts / 不同微积分概念的不同证明技术
- **Decision Points / 决策点**: Key decision points in proof construction / 证明构造中的关键决策点
- **Multiple Approaches / 多种方法**: Multiple proof approaches for the same theorem / 同一定理的多种证明方法

---

## 2. Limit Proof Decision Trees / 极限证明决策树

### 2.1 Limit Existence Decision Tree / 极限存在性决策树

```mermaid
flowchart TD
    Start[Prove Limit Exists<br/>证明极限存在<br/>lim_{x→a} f(x) = L] --> Q1{Function Type?<br/>函数类型?}

    Q1 -->|Polynomial<br/>多项式| Direct[Direct Substitution<br/>直接代入<br/>lim x^n = a^n]
    Q1 -->|Rational<br/>有理函数| Rational{Indeterminate?<br/>不定式?}
    Rational -->|0/0| LHopital[L'Hôpital's Rule<br/>洛必达法则<br/>lim f/g = lim f'/g']
    Rational -->|∞/∞| LHopital
    Rational -->|Factorable| Factor[Factor and Cancel<br/>因式分解并约分]

    Q1 -->|Trigonometric<br/>三角| Trig{Standard Limit?<br/>标准极限?}
    Trig -->|sin(x)/x| Standard[Use lim sin(x)/x = 1<br/>使用lim sin(x)/x = 1]
    Trig -->|Other| Transform[Transform to Standard<br/>转换为标准形式]

    Q1 -->|Exponential<br/>指数| Exp[Use Continuity<br/>使用连续性<br/>lim e^x = e^a]
    Q1 -->|Piecewise<br/>分段| Piece[Check Left and Right<br/>检查左右极限<br/>lim^- = lim^+]

    Q1 -->|General<br/>一般| EpsilonDelta[ε-δ Definition<br/>ε-δ定义<br/>For any ε, find δ]

    style Start fill:#e1f5ff
    style Direct fill:#c8e6c9
    style LHopital fill:#c8e6c9
    style Standard fill:#c8e6c9
    style Exp fill:#c8e6c9
    style EpsilonDelta fill:#fff4e1
```

### 2.2 Limit Uniqueness Decision Tree / 极限唯一性决策树

```mermaid
flowchart TD
    Start[Prove Limit Unique<br/>证明极限唯一] --> Q1{Assume Two Limits<br/>假设两个极限<br/>L₁ and L₂}

    Q1 --> Step1[Use ε-δ Definition<br/>使用ε-δ定义<br/>For L₁ and L₂]
    Step1 --> Step2[Choose ε = |L₁ - L₂|/2<br/>选择ε = |L₁ - L₂|/2]
    Step2 --> Step3[Find δ for both<br/>为两者找到δ<br/>|x-a| < δ]
    Step3 --> Step4[Apply Triangle Inequality<br/>应用三角不等式<br/>|L₁ - L₂| < ε]
    Step4 --> Contradiction[Contradiction<br/>矛盾<br/>|L₁ - L₂| < |L₁ - L₂|/2]
    Contradiction --> Result[L₁ = L₂ ✓]

    style Start fill:#e1f5ff
    style Result fill:#c8e6c9
```

---

## 3. Derivative Proof Decision Trees / 导数证明决策树

### 3.1 Differentiability Proof Decision Tree / 可微性证明决策树

```mermaid
flowchart TD
    Start[Prove f is Differentiable<br/>证明f可微<br/>at point a] --> Q1{Definition Method?<br/>定义方法?}

    Q1 -->|Limit Definition<br/>极限定义| LimitDef[Use f'(a) = lim_{h→0} [f(a+h)-f(a)]/h<br/>使用f'(a) = lim_{h→0} [f(a+h)-f(a)]/h]
    Q1 -->|Increment Definition<br/>增量定义| IncrementDef[Use f'(a) = lim_{Δx→0} Δy/Δx<br/>使用f'(a) = lim_{Δx→0} Δy/Δx]
    Q1 -->|Tangent Line<br/>切线| Tangent[Show Tangent Line Exists<br/>显示切线存在<br/>y = f(a) + f'(a)(x-a)]

    LimitDef --> Q2{Limit Exists?<br/>极限存在?}
    Q2 -->|Yes| Result1[f is Differentiable ✓]
    Q2 -->|No| CheckCont[Check Continuity<br/>检查连续性<br/>Differentiable implies continuous]

    style Start fill:#e1f5ff
    style Result1 fill:#c8e6c9
```

### 3.2 Chain Rule Proof Decision Tree / 链式法则证明决策树

```mermaid
flowchart TD
    Start[Prove Chain Rule<br/>证明链式法则<br/>D(g∘f) = (Dg∘f)·Df] --> Q1{Proof Method?<br/>证明方法?}

    Q1 -->|Limit Definition<br/>极限定义| LimitMethod[Use Limit Definition<br/>使用极限定义<br/>lim_{h→0} [g(f(x+h))-g(f(x))]/h]
    Q1 -->|Increment Method<br/>增量方法| IncrementMethod[Use Δy = g'(f(x))Δf + o(Δf)<br/>使用Δy = g'(f(x))Δf + o(Δf)]
    Q1 -->|Category Theory<br/>范畴论| CategoryMethod[Use Functoriality<br/>使用函子性<br/>D preserves composition]

    LimitMethod --> Step1[Let k = f(x+h) - f(x)<br/>设k = f(x+h) - f(x)]
    Step1 --> Step2[Rewrite as [g(f(x)+k)-g(f(x))]/k · k/h<br/>重写为[g(f(x)+k)-g(f(x))]/k · k/h]
    Step2 --> Step3[Take Limits<br/>取极限<br/>g'(f(x)) · f'(x)]
    Step3 --> Result[Chain Rule ✓]

    style Start fill:#e1f5ff
    style Result fill:#c8e6c9
```

---

## 4. Integral Proof Decision Trees / 积分证明决策树

### 4.1 Integrability Proof Decision Tree / 可积性证明决策树

```mermaid
flowchart TD
    Start[Prove f is Integrable<br/>证明f可积<br/>on [a,b]] --> Q1{Integral Type?<br/>积分类型?}

    Q1 -->|Riemann<br/>黎曼| Riemann{Continuous?<br/>连续?}
    Riemann -->|Yes| ContThm[Continuous implies Riemann Integrable<br/>连续蕴含黎曼可积<br/>Use Uniform Continuity]
    Riemann -->|No| Bounded{Bounded with<br/>Finitely Many<br/>Discontinuities?<br/>有界且有限个<br/>不连续点?}
    Bounded -->|Yes| BoundedThm[Bounded with Finitely Many<br/>Discontinuities Implies Integrable<br/>有界且有限个不连续点蕴含可积]
    Bounded -->|No| CheckLebesgue[Check Lebesgue Integrable<br/>检查Lebesgue可积]

    Q1 -->|Lebesgue<br/>Lebesgue| Lebesgue{Measure Zero<br/>Discontinuities?<br/>测度零<br/>不连续点?}
    Lebesgue -->|Yes| LebesgueThm[Lebesgue Integrable<br/>Lebesgue可积]
    Lebesgue -->|No| CheckRiemann[Check Riemann Integrable<br/>检查黎曼可积]

    style Start fill:#e1f5ff
    style ContThm fill:#c8e6c9
    style BoundedThm fill:#c8e6c9
    style LebesgueThm fill:#c8e6c9
```

### 4.2 Fundamental Theorem Proof Decision Tree / 微积分基本定理证明决策树

```mermaid
flowchart TD
    Start[Prove Fundamental Theorem<br/>证明微积分基本定理<br/>D∘I = id] --> Q1{Part?<br/>部分?}

    Q1 -->|Part I<br/>第一部分| Part1[Prove D(I(f)) = f<br/>证明D(I(f)) = f]
    Q1 -->|Part II<br/>第二部分| Part2[Prove I(D(f)) = f - f(a)<br/>证明I(D(f)) = f - f(a)]

    Part1 --> Step1[Use Definition<br/>使用定义<br/>D(I(f))(x) = lim_{h→0} [I(f)(x+h)-I(f)(x)]/h]
    Step1 --> Step2[Simplify Integral<br/>简化积分<br/>= lim_{h→0} [∫_x^{x+h} f(t)dt]/h]
    Step2 --> Step3[Use Mean Value Theorem<br/>使用中值定理<br/>= lim_{h→0} f(c)·h/h]
    Step3 --> Result1[f(x) ✓]

    Part2 --> Step4[Use Definition<br/>使用定义<br/>I(D(f))(x) = ∫_a^x f'(t)dt]
    Step4 --> Step5[Apply FTC Part I<br/>应用FTC第一部分<br/>= f(x) - f(a)]
    Step5 --> Result2[f(x) - f(a) ✓]

    style Start fill:#e1f5ff
    style Result1 fill:#c8e6c9
    style Result2 fill:#c8e6c9
```

---

## 5. Series Proof Decision Trees / 级数证明决策树

### 5.1 Convergence Proof Decision Tree / 收敛性证明决策树

```mermaid
flowchart TD
    Start[Prove Series Converges<br/>证明级数收敛<br/>∑a_n] --> Q1{Test Type?<br/>测试类型?}

    Q1 -->|Ratio Test<br/>比值判别法| Ratio{lim |a_{n+1}/a_n| < 1?<br/>lim |a_{n+1}/a_n| < 1?}
    Ratio -->|Yes| RatioConv[Converges ✓]
    Ratio -->|No| RatioDiv{> 1?<br/>> 1?}
    RatioDiv -->|Yes| RatioDiverges[Diverges ✗]
    RatioDiv -->|No| TryOther[Try Other Test<br/>尝试其他测试]

    Q1 -->|Root Test<br/>根值判别法| Root{lim |a_n|^{1/n} < 1?<br/>lim |a_n|^{1/n} < 1?}
    Root -->|Yes| RootConv[Converges ✓]
    Root -->|No| RootDiv{> 1?<br/>> 1?}
    RootDiv -->|Yes| RootDiverges[Diverges ✗]
    RootDiv -->|No| TryOther

    Q1 -->|Comparison Test<br/>比较判别法| Compare{Find b_n with<br/>Known Convergence<br/>找到b_n具有<br/>已知收敛性}
    Compare -->|Yes| CompareConv[If ∑b_n converges and<br/>|a_n| ≤ b_n, then ∑a_n converges ✓]
    Compare -->|No| TryOther

    Q1 -->|Integral Test<br/>积分判别法| Integral{∫_1^∞ f(x)dx<br/>Converges?<br/>∫_1^∞ f(x)dx<br/>收敛?}
    Integral -->|Yes| IntegralConv[Series Converges ✓]
    Integral -->|No| IntegralDiv[Series Diverges ✗]

    style Start fill:#e1f5ff
    style RatioConv fill:#c8e6c9
    style RootConv fill:#c8e6c9
    style CompareConv fill:#c8e6c9
    style IntegralConv fill:#c8e6c9
```

---

## 6. Proof Strategy Selection / 证明策略选择

### 6.1 When to Use Direct Construction / 何时使用直接构造

**Use for / 用于**:

- Limit existence (polynomial, rational functions)
- Derivative computation (basic functions)
- Integral computation (basic antiderivatives)

**Example / 例子**: $\lim_{x \to 2} x^2 = 4$ (direct substitution)

### 6.2 When to Use ε-δ Definition / 何时使用ε-δ定义

**Use for / 用于**:

- Limit existence (general functions)
- Continuity proofs
- Uniqueness proofs

**Example / 例子**: $\lim_{x \to 0} \frac{\sin x}{x} = 1$ (ε-δ proof)

### 6.3 When to Use Mean Value Theorem / 何时使用中值定理

**Use for / 用于**:

- Fundamental Theorem of Calculus
- Derivative properties
- Integral properties

**Example / 例子**: Fundamental Theorem Part I proof

### 6.4 When to Use Category Theory / 何时使用范畴论

**Use for / 用于**:

- Functoriality proofs
- Natural transformation proofs
- Universal property proofs

**Example / 例子**: Chain rule as functoriality of derivative functor

---

## 7. Detailed Proof Examples / 详细证明示例

### Example 1: Limit Existence Proof / 例子1：极限存在性证明

**Problem / 问题**: Prove $\lim_{x \to 0} \frac{\sin x}{x} = 1$

**Decision Path / 决策路径**:

1. Function type? → Trigonometric
2. Standard limit? → Yes (fundamental limit)
3. Use geometric proof or L'Hôpital's rule

**Proof Steps / 证明步骤**:

**Method 1: Geometric Proof / 方法1：几何证明**

1. Consider unit circle with angle $x$
2. Area of triangle < Area of sector < Area of larger triangle
3. $\frac{1}{2}\sin x < \frac{1}{2}x < \frac{1}{2}\tan x$
4. Divide by $\sin x$: $1 < \frac{x}{\sin x} < \frac{1}{\cos x}$
5. Take reciprocals: $\cos x < \frac{\sin x}{x} < 1$
6. Apply squeeze theorem: $\lim_{x \to 0} \frac{\sin x}{x} = 1$ ✓

**Method 2: L'Hôpital's Rule / 方法2：洛必达法则**

1. Form: $\frac{0}{0}$ (indeterminate)
2. Apply L'Hôpital: $\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1$ ✓

### Example 2: Chain Rule Proof / 例子2：链式法则证明

**Problem / 问题**: Prove $(g \circ f)'(x) = g'(f(x)) \cdot f'(x)$

**Decision Path / 决策路径**:

1. Proof method? → Limit definition
2. Use increment method with key trick

**Proof Steps / 证明步骤**:

1. **Definition / 定义**:
   $$(g \circ f)'(x) = \lim_{h \to 0} \frac{g(f(x+h)) - g(f(x))}{h}$$

2. **Key Trick / 关键技巧**: Let $k = f(x+h) - f(x)$

3. **Rewrite / 重写**:
   $$= \lim_{h \to 0} \frac{g(f(x) + k) - g(f(x))}{k} \cdot \frac{k}{h}$$

4. **Apply Limits / 应用极限**:
   $$= \lim_{k \to 0} \frac{g(f(x) + k) - g(f(x))}{k} \cdot \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

5. **Result / 结果**:
   $$= g'(f(x)) \cdot f'(x) \quad \square$$

### Example 3: Fundamental Theorem Proof / 例子3：微积分基本定理证明

**Problem / 问题**: Prove $D(I(f))(x) = f(x)$ (FTC Part I)

**Decision Path / 决策路径**:

1. Use definition of derivative
2. Use Mean Value Theorem for integrals

**Proof Steps / 证明步骤**:

1. **Definition / 定义**:
   $$D(I(f))(x) = \lim_{h \to 0} \frac{\int_a^{x+h} f(t) dt - \int_a^x f(t) dt}{h}$$

2. **Simplify / 化简**:
   $$= \lim_{h \to 0} \frac{\int_x^{x+h} f(t) dt}{h}$$

3. **Mean Value Theorem / 中值定理**: There exists $c \in [x, x+h]$ such that:
   $$\int_x^{x+h} f(t) dt = f(c) \cdot h$$

4. **Apply Limit / 应用极限**:
   $$= \lim_{h \to 0} \frac{f(c) \cdot h}{h} = \lim_{h \to 0} f(c)$$

5. **Continuity / 连续性**: As $h \to 0$, $c \to x$, so:
   $$= f(x) \quad \square$$

---

## 8. References / 参考文献

### 8.1 Mathematical References / 数学参考文献

**Standard Calculus Textbooks / 标准微积分教材**:

- **Apostol, T. M.** (1967). *Calculus, Volume 1* (2nd ed.). Wiley. - Standard reference / 标准参考
- **Spivak, M.** (2008). *Calculus* (4th ed.). Publish or Perish. - Rigorous approach / 严格方法
- **Stewart, J.** (2020). *Calculus: Early Transcendentals* (8th ed.). Cengage Learning. - Comprehensive / 全面

**Note / 注意**: These are established textbooks. Check for latest editions and supplementary materials. / 这些是已确立的教材。请检查最新版本和补充材料。

### 8.2 International Standards / 国际标准

**Note / 注意**: Proof techniques are covered in all standard calculus courses. The following are general references. / 证明技术在所有标准微积分课程中都有涵盖。以下是一般参考。

**Calculus Courses / 微积分课程**:

- **MIT 18.01, 18.02**: Single and multivariable calculus (foundational)
- **Harvard Math 1A, Math 21a**: Calculus courses
- **Stanford MATH19, MATH51**: Calculus courses

### 8.3 Related Files / 相关文件

- `resource/Category/10-Proof-Trees/01-Axiom-Theorem-Networks/01-Calculus-Networks.md` - Calculus axiom-theorem networks
- `resource/Category/10-Proof-Trees/03-Proof-Networks/` - Detailed proof networks
- `resource/Concept/01-微积分基础/01-极限的多种视角.md` - Multiple perspectives on limits

---

**Last Updated / 最后更新**: 2026-01-16
**Standards / 标准**: 2026-2027 Enhanced Cross-Disciplinary Standard
**Status / 状态**: ✅ Complete with cognitive representations, proof networks, and multiple perspectives / 完成，包含认知表征、证明网络和多重视角

### **2026-2027框架对齐说明 / 2026-2027 Framework Alignment**

本文件已对齐以下2026-2027最新框架：

- **多重认知表征**：Mermaid流程图、决策树、证明网络，激活不同认知通道
- **多重视角解释**：直接构造、ε-δ定义、中值定理、范畴论视角，提供直观理解
- **完整证明网络**：极限、导数、积分、级数的分步证明
- **公理-定理网络**：从实数公理到应用的完整逻辑依赖关系
- **国际标准**：使用实际存在的微积分课程和教材
