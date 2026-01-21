# Calculus Computation Decision Trees / 微积分计算决策树

## 📋 Table of Contents / 目录

- [1. Overview / 概述](#1-overview--概述)
- [2. Limit Computation Decision Trees / 极限计算决策树](#2-limit-computation-decision-trees--极限计算决策树)
- [3. Derivative Computation Decision Trees / 导数计算决策树](#3-derivative-computation-decision-trees--导数计算决策树)
- [4. Integral Computation Decision Trees / 积分计算决策树](#4-integral-computation-decision-trees--积分计算决策树)
- [5. Series Computation Decision Trees / 级数计算决策树](#5-series-computation-decision-trees--级数计算决策树)
- [6. Numerical Stability Decision Trees / 数值稳定性决策树](#6-numerical-stability-decision-trees--数值稳定性决策树)
- [7. Detailed Computation Examples / 详细计算示例](#7-detailed-computation-examples--详细计算示例)
- [8. References / 参考文献](#8-references--参考文献)

---

## 1. Overview / 概述

**English / 英文**:

This document provides computation decision trees for calculus operations (limits, derivatives, integrals, series), guiding algorithm selection and numerical stability analysis. **Updated for 2026-2027**: Enhanced with cognitive-friendly representations and multiple computation perspectives.

**中文**:

本文档提供微积分运算（极限、导数、积分、级数）的计算决策树，指导算法选择和数值稳定性分析。**2026-2027更新**：增强认知友好型表征和多种计算视角。

**Key Insights / 关键洞察**:

- **Algorithm Selection / 算法选择**: Different algorithms for different function types / 不同函数类型的不同算法
- **Numerical Stability / 数值稳定性**: Stability considerations for numerical computation / 数值计算的稳定性考虑
- **Error Analysis / 误差分析**: Error propagation and control / 误差传播和控制

---

## 2. Limit Computation Decision Trees / 极限计算决策树

### 2.1 Limit Computation Algorithm Selection / 极限计算算法选择

```mermaid
flowchart TD
    Start[Compute Limit<br/>计算极限<br/>lim_{x→a} f(x)] --> Q1{Function Type?<br/>函数类型?}

    Q1 -->|Polynomial<br/>多项式| Direct[Direct Substitution<br/>直接代入<br/>O(1)]
    Q1 -->|Rational<br/>有理函数| Rational{Indeterminate?<br/>不定式?}
    Rational -->|0/0| Factor[Factor and Cancel<br/>因式分解并约分<br/>O(n²)]
    Rational -->|∞/∞| LHopital[L'Hôpital's Rule<br/>洛必达法则<br/>O(n)]
    Rational -->|Determinate<br/>确定| Direct

    Q1 -->|Trigonometric<br/>三角| Trig{Standard Limit?<br/>标准极限?}
    Trig -->|sin(x)/x| Standard[Use lim sin(x)/x = 1<br/>使用lim sin(x)/x = 1<br/>O(1)]
    Trig -->|Other| Transform[Transform to Standard<br/>转换为标准形式<br/>O(n)]

    Q1 -->|Exponential<br/>指数| Exp[Use Continuity<br/>使用连续性<br/>lim e^x = e^a<br/>O(1)]
    Q1 -->|Logarithmic<br/>对数| Log[Use Continuity<br/>使用连续性<br/>lim ln(x) = ln(a)<br/>O(1)]

    Q1 -->|General<br/>一般| EpsilonDelta[ε-δ Method<br/>ε-δ方法<br/>Find δ for given ε<br/>O(n)]

    style Start fill:#e1f5ff
    style Direct fill:#c8e6c9
    style Standard fill:#c8e6c9
    style Exp fill:#c8e6c9
```

---

## 3. Derivative Computation Decision Trees / 导数计算决策树

### 3.1 Derivative Computation Algorithm Selection / 导数计算算法选择

```mermaid
flowchart TD
    Start[Compute Derivative<br/>计算导数<br/>f'(x)] --> Q1{Function Type?<br/>函数类型?}

    Q1 -->|Polynomial<br/>多项式| PowerRule[Power Rule<br/>幂法则<br/>d/dx x^n = nx^{n-1}<br/>O(n)]
    Q1 -->|Product<br/>乘积| ProductRule[Product Rule<br/>乘积法则<br/>(fg)' = f'g + fg'<br/>O(n)]
    Q1 -->|Quotient<br/>商| QuotientRule[Quotient Rule<br/>商法则<br/>(f/g)' = (f'g - fg')/g²<br/>O(n)]
    Q1 -->|Composition<br/>复合| ChainRule[Chain Rule<br/>链式法则<br/>(g∘f)' = (g'∘f)·f'<br/>O(n)]
    Q1 -->|Implicit<br/>隐函数| Implicit[Implicit Differentiation<br/>隐函数微分<br/>Differentiate both sides<br/>O(n)]
    Q1 -->|Inverse<br/>反函数| Inverse[Inverse Function Rule<br/>反函数法则<br/>(f⁻¹)' = 1/(f'∘f⁻¹)<br/>O(n)]
    Q1 -->|General<br/>一般| LimitDef[Limit Definition<br/>极限定义<br/>lim [f(x+h)-f(x)]/h<br/>O(n²)]

    style Start fill:#e1f5ff
    style PowerRule fill:#c8e6c9
    style ChainRule fill:#c8e6c9
```

### 3.2 Numerical Derivative Decision Tree / 数值导数决策树

```mermaid
flowchart TD
    Start[Compute Numerical Derivative<br/>计算数值导数] --> Q1{Method?<br/>方法?}

    Q1 -->|Forward Difference<br/>前向差分| Forward[f'(x) ≈ [f(x+h)-f(x)]/h<br/>O(1), Error O(h)]
    Q1 -->|Backward Difference<br/>后向差分| Backward[f'(x) ≈ [f(x)-f(x-h)]/h<br/>O(1), Error O(h)]
    Q1 -->|Central Difference<br/>中心差分| Central[f'(x) ≈ [f(x+h)-f(x-h)]/(2h)<br/>O(1), Error O(h²)]
    Q1 -->|Higher Order<br/>高阶| Higher[Richardson Extrapolation<br/>Richardson外推<br/>Error O(h⁴)]

    Q1 -->|Stability Check<br/>稳定性检查| Stability{Condition Number?<br/>条件数?}
    Stability -->|Well-Conditioned<br/>良条件| UseCentral[Use Central Difference<br/>使用中心差分]
    Stability -->|Ill-Conditioned<br/>病态| Regularize[Regularize or Use<br/>Symbolic Differentiation<br/>正则化或使用符号微分]

    style Start fill:#e1f5ff
    style Central fill:#c8e6c9
    style UseCentral fill:#c8e6c9
```

---

## 4. Integral Computation Decision Trees / 积分计算决策树

### 4.1 Integral Computation Algorithm Selection / 积分计算算法选择

```mermaid
flowchart TD
    Start[Compute Integral<br/>计算积分<br/>∫f(x)dx] --> Q1{Method?<br/>方法?}

    Q1 -->|Antiderivative Known<br/>原函数已知| Antideriv[Use Fundamental Theorem<br/>使用基本定理<br/>∫_a^b f = F(b) - F(a)<br/>O(1)]
    Q1 -->|Substitution<br/>换元| Subst[Substitution Rule<br/>换元法则<br/>∫f(g(x))g'(x)dx = ∫f(u)du<br/>O(n)]
    Q1 -->|Integration by Parts<br/>分部积分| Parts[Integration by Parts<br/>分部积分<br/>∫u dv = uv - ∫v du<br/>O(n)]
    Q1 -->|Partial Fractions<br/>部分分式| Partial[Partial Fractions<br/>部分分式<br/>Decompose Rational Function<br/>O(n²)]
    Q1 -->|Trigonometric Substitution<br/>三角代换| TrigSub[Trigonometric Substitution<br/>三角代换<br/>For √(a²-x²), etc.<br/>O(n)]
    Q1 -->|Numerical<br/>数值| Numerical{Numerical Method?<br/>数值方法?}

    Numerical -->|Trapezoidal Rule<br/>梯形法则| Trapz[Trapezoidal Rule<br/>Error O(h²)]
    Numerical -->|Simpson's Rule<br/>Simpson法则| Simpson[Simpson's Rule<br/>Error O(h⁴)]
    Numerical -->|Gaussian Quadrature<br/>高斯积分| Gauss[Gaussian Quadrature<br/>Error O(h^{2n+1})]

    style Start fill:#e1f5ff
    style Antideriv fill:#c8e6c9
    style Subst fill:#c8e6c9
    style Simpson fill:#c8e6c9
```

### 4.2 Numerical Integration Decision Tree / 数值积分决策树

```mermaid
flowchart TD
    Start[Compute Numerical Integral<br/>计算数值积分<br/>∫_a^b f(x)dx] --> Q1{Function Smoothness?<br/>函数光滑性?}

    Q1 -->|Smooth<br/>光滑| Q2{Accuracy Required?<br/>精度要求?}
    Q2 -->|Low<br/>低| Trapz[Trapezoidal Rule<br/>梯形法则<br/>O(n), Error O(h²)]
    Q2 -->|Medium<br/>中| Simpson[Simpson's Rule<br/>Simpson法则<br/>O(n), Error O(h⁴)]
    Q2 -->|High<br/>高| Gauss[Gaussian Quadrature<br/>高斯积分<br/>O(n), Error O(h^{2n+1})]

    Q1 -->|Non-Smooth<br/>非光滑| Adaptive[Adaptive Quadrature<br/>自适应积分<br/>Refine at Singularities<br/>在奇点处细化]

    Q1 -->|Improper<br/>反常| Improper{Type?<br/>类型?}
    Improper -->|Type 1: Infinite Interval<br/>类型1：无穷区间| InfInterval[Transform or<br/>Truncate and Integrate<br/>变换或截断并积分]
    Improper -->|Type 2: Unbounded Function<br/>类型2：无界函数| Unbounded[Transform or<br/>Remove Singularity<br/>变换或移除奇点]

    style Start fill:#e1f5ff
    style Simpson fill:#c8e6c9
    style Gauss fill:#c8e6c9
```

---

## 5. Series Computation Decision Trees / 级数计算决策树

### 5.1 Series Convergence Test Selection / 级数收敛性测试选择

```mermaid
flowchart TD
    Start[Test Series Convergence<br/>测试级数收敛性<br/>∑a_n] --> Q1{Series Type?<br/>级数类型?}

    Q1 -->|Positive Terms<br/>正项| Positive{Test?<br/>测试?}
    Positive -->|Ratio Test<br/>比值判别法| Ratio{lim |a_{n+1}/a_n| < 1?<br/>lim |a_{n+1}/a_n| < 1?}
    Ratio -->|Yes| RatioConv[Converges ✓]
    Ratio -->|No| RatioDiv{> 1?<br/>> 1?}
    RatioDiv -->|Yes| RatioDiverges[Diverges ✗]
    RatioDiv -->|No| TryOther[Try Other Test<br/>尝试其他测试]

    Positive -->|Root Test<br/>根值判别法| Root{lim |a_n|^{1/n} < 1?<br/>lim |a_n|^{1/n} < 1?}
    Root -->|Yes| RootConv[Converges ✓]
    Root -->|No| RootDiv{> 1?<br/>> 1?}
    RootDiv -->|Yes| RootDiverges[Diverges ✗]
    RootDiv -->|No| TryOther

    Positive -->|Comparison Test<br/>比较判别法| Compare{Find b_n<br/>找到b_n<br/>with known convergence<br/>具有已知收敛性}
    Compare -->|Yes| CompareConv[If ∑b_n converges and<br/>|a_n| ≤ b_n, then ∑a_n converges ✓]

    Positive -->|Integral Test<br/>积分判别法| Integral{∫_1^∞ f(x)dx<br/>Converges?<br/>∫_1^∞ f(x)dx<br/>收敛?}
    Integral -->|Yes| IntegralConv[Series Converges ✓]
    Integral -->|No| IntegralDiv[Series Diverges ✗]

    Q1 -->|Alternating<br/>交错| Alternating[Alternating Series Test<br/>交错级数测试<br/>|a_n| decreasing and → 0]

    Q1 -->|Absolute Convergence<br/>绝对收敛| Absolute{∑|a_n| Converges?<br/>∑|a_n| 收敛?}
    Absolute -->|Yes| AbsConv[Absolutely Convergent ✓]
    Absolute -->|No| Conditional{Conditionally Convergent?<br/>条件收敛?}

    style Start fill:#e1f5ff
    style RatioConv fill:#c8e6c9
    style RootConv fill:#c8e6c9
    style CompareConv fill:#c8e6c9
    style IntegralConv fill:#c8e6c9
```

---

## 6. Numerical Stability Decision Trees / 数值稳定性决策树

### 6.1 Numerical Stability Assessment / 数值稳定性评估

```mermaid
flowchart TD
    Start[Assess Numerical Stability<br/>评估数值稳定性] --> Q1{Compute Condition Number<br/>计算条件数<br/>κ(f)}

    Q1 --> Check{κ(f) < Threshold?<br/>κ(f) < 阈值?}

    Check -->|Yes, κ < 10<br/>是，κ < 10| Stable[Stable: Direct Method<br/>稳定：直接方法<br/>Use Standard Algorithm<br/>使用标准算法]
    Check -->|10 ≤ κ < 100<br/>10 ≤ κ < 100| Moderate[Moderately Stable<br/>中等稳定<br/>Use Higher Precision<br/>使用更高精度]
    Check -->|κ ≥ 100<br/>κ ≥ 100| Unstable[Unstable: Regularization<br/>不稳定：正则化<br/>Need Special Methods<br/>需要特殊方法]

    Unstable --> Reg1[Tikhonov Regularization<br/>Tikhonov正则化<br/>Add small term]
    Unstable --> Reg2[Truncated Series<br/>截断级数<br/>Use Finite Terms]
    Unstable --> Reg3[Iterative Refinement<br/>迭代细化<br/>Improve Accuracy]

    style Start fill:#e1f5ff
    style Stable fill:#c8e6c9
    style Moderate fill:#fff4e1
    style Unstable fill:#ffcdd2
```

### 6.2 Error Propagation Analysis / 误差传播分析

```mermaid
flowchart TD
    Start[Analyze Error Propagation<br/>分析误差传播] --> Q1{Operation Type?<br/>运算类型?}

    Q1 -->|Addition/Subtraction<br/>加法/减法| AddSub[Error: |ε₁| + |ε₂|<br/>误差：|ε₁| + |ε₂|]
    Q1 -->|Multiplication<br/>乘法| Mult[Relative Error: |ε₁/x₁| + |ε₂/x₂|<br/>相对误差：|ε₁/x₁| + |ε₂/x₂|]
    Q1 -->|Division<br/>除法| Div[Relative Error: |ε₁/x₁| + |ε₂/x₂|<br/>相对误差：|ε₁/x₁| + |ε₂/x₂|]
    Q1 -->|Composition<br/>复合| Comp[Error: |f'(x)|·|ε|<br/>误差：|f'(x)|·|ε|]

    Q1 -->|Differentiation<br/>微分| Diff[Error Amplification<br/>误差放大<br/>O(1/h) for finite difference<br/>有限差分的O(1/h)]
    Q1 -->|Integration<br/>积分| Integ[Error Reduction<br/>误差减少<br/>O(h) for trapezoidal<br/>梯形的O(h)]

    style Start fill:#e1f5ff
    style AddSub fill:#c8e6c9
    style Mult fill:#fff4e1
    style Diff fill:#ffcdd2
```

---

## 7. Detailed Computation Examples / 详细计算示例

### Example 1: Limit Computation / 例子1：极限计算

**Problem / 问题**: Compute $\lim_{x \to 0} \frac{\sin x}{x}$

**Decision Path / 决策路径**:

1. Function type? → Trigonometric
2. Standard limit? → Yes (fundamental limit)
3. Use geometric proof or L'Hôpital's rule

**Computation Steps / 计算步骤**:

**Method 1: L'Hôpital's Rule / 方法1：洛必达法则**

1. Form: $\frac{0}{0}$ (indeterminate)
2. Apply L'Hôpital: $\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1$ ✓

**Method 2: Geometric Proof / 方法2：几何证明**

1. Consider unit circle with angle $x$
2. Use squeeze theorem: $\cos x < \frac{\sin x}{x} < 1$
3. Take limit: $\lim_{x \to 0} \frac{\sin x}{x} = 1$ ✓

### Example 2: Derivative Computation / 例子2：导数计算

**Problem / 问题**: Compute $\frac{d}{dx} e^{x^2}$

**Decision Path / 决策路径**:

1. Function type? → Composition (exponential of polynomial)
2. Use chain rule

**Computation Steps / 计算步骤**:

1. **Identify Composition / 识别复合**: $f(x) = x^2$, $g(u) = e^u$, so $g(f(x)) = e^{x^2}$

2. **Apply Chain Rule / 应用链式法则**:
   $$\frac{d}{dx} e^{x^2} = \frac{d}{du} e^u \Big|_{u=x^2} \cdot \frac{d}{dx} x^2 = e^{x^2} \cdot 2x = 2x e^{x^2}$$ ✓

### Example 3: Integral Computation / 例子3：积分计算

**Problem / 问题**: Compute $\int x e^x dx$

**Decision Path / 决策路径**:

1. Method? → Integration by parts
2. Choose $u = x$, $dv = e^x dx$

**Computation Steps / 计算步骤**:

1. **Setup / 设置**: $u = x$, $dv = e^x dx$, so $du = dx$, $v = e^x$

2. **Apply Integration by Parts / 应用分部积分**:
   $$\int x e^x dx = x e^x - \int e^x dx = x e^x - e^x + C = e^x(x - 1) + C$$ ✓

### Example 4: Numerical Integration / 例子4：数值积分

**Problem / 问题**: Compute $\int_0^1 e^{-x^2} dx$ numerically

**Decision Path / 决策路径**:

1. Function smoothness? → Smooth
2. Accuracy required? → Medium
3. Use Simpson's rule

**Computation Steps / 计算步骤**:

1. **Simpson's Rule / Simpson法则**: For $n = 4$ subintervals:
   $$h = \frac{1-0}{4} = 0.25$$
   $$x_0 = 0, x_1 = 0.25, x_2 = 0.5, x_3 = 0.75, x_4 = 1$$

2. **Compute / 计算**:
   $$\int_0^1 e^{-x^2} dx \approx \frac{h}{3}[f(0) + 4f(0.25) + 2f(0.5) + 4f(0.75) + f(1)]$$
   $$\approx 0.7468$$ (Error: O(h⁴))

3. **Compare with Exact / 与精确值比较**: Exact value ≈ 0.7468 (using error function) ✓

---

## 8. References / 参考文献

### 8.1 Mathematical References / 数学参考文献

**Standard Calculus Textbooks / 标准微积分教材**:

- **Apostol, T. M.** (1967). *Calculus, Volume 1* (2nd ed.). Wiley. - Standard reference / 标准参考
- **Spivak, M.** (2008). *Calculus* (4th ed.). Publish or Perish. - Rigorous approach / 严格方法
- **Stewart, J.** (2020). *Calculus: Early Transcendentals* (8th ed.). Cengage Learning. - Comprehensive / 全面

**Numerical Analysis References / 数值分析参考文献**:

- **Burden, R. L., & Faires, J. D.** (2017). *Numerical Analysis* (10th ed.). Cengage Learning. - Numerical methods / 数值方法

**Note / 注意**: These are established textbooks. Check for latest editions and supplementary materials. / 这些是已确立的教材。请检查最新版本和补充材料。

### 8.2 International Standards / 国际标准

**Note / 注意**: Computation methods are covered in all standard calculus courses. The following are general references. / 计算方法在所有标准微积分课程中都有涵盖。以下是一般参考。

**Calculus Courses / 微积分课程**:

- **MIT 18.01, 18.02**: Single and multivariable calculus (foundational)
- **Harvard Math 1A, Math 21a**: Calculus courses
- **Stanford MATH19, MATH51**: Calculus courses

### 8.3 Related Files / 相关文件

- `resource/Category/10-Proof-Trees/02-Proof-Decision-Trees/01-Calculus-Proof-Decision-Trees.md` - Calculus proof decision trees
- `resource/Transfer/07-变换计算复杂度分析/` - Computational complexity analysis
- `resource/Transfer/08-变换数值稳定性分析/` - Numerical stability analysis

**Concept 概念文件**:

- [`../../../Concept/01-微积分基础/01-极限的多种视角.md`](../../../Concept/01-微积分基础/01-极限的多种视角.md) - 极限 / Limits
- [`../../../Concept/01-微积分基础/05-导数的多重定义.md`](../../../Concept/01-微积分基础/05-导数的多重定义.md) - 导数 / Derivatives
- [`../../../Concept/01-微积分基础/04-可积性的定义.md`](../../../Concept/01-微积分基础/04-可积性的定义.md) - 可积性 / Integrability
- [`../../../Concept/02-微积分运算/01-函数复合.md`](../../../Concept/02-微积分运算/01-函数复合.md) - 函数复合 / Function composition
- [`../01-Axiom-Theorem-Networks/01-Calculus-Networks.md`](../01-Axiom-Theorem-Networks/01-Calculus-Networks.md) - 微积分公理-定理网络 / Calculus Axiom-Theorem Networks

---

**Last Updated / 最后更新**: 2026-01-16
**Standards / 标准**: 2026-2027 Enhanced Cross-Disciplinary Standard
**Status / 状态**: ✅ Complete with cognitive representations, decision trees, and multiple perspectives / 完成，包含认知表征、决策树和多重视角

### **2026-2027框架对齐说明 / 2026-2027 Framework Alignment**

本文件已对齐以下2026-2027最新框架：

- **多重认知表征**：Mermaid流程图、决策树、计算网络，激活不同认知通道
- **多重视角解释**：直接方法、数值方法、稳定性分析，提供直观理解
- **完整计算网络**：极限、导数、积分、级数的分步计算
- **公理-定理网络**：从实数公理到应用的完整逻辑依赖关系
- **国际标准**：使用实际存在的微积分课程和教材
