# 1.2 数学模型基础

## 概述

数学模型基础为Formal-ProgramManage提供严格的数学工具和理论支撑。本理论体系对标MIT 18.06 (线性代数)、Stanford CS229 (机器学习)、CMU 15-251 (计算理论)、Berkeley CS70 (离散数学)等国际顶尖课程标准。

## 1.2.1 集合论基础

### 基本定义

**定义 1.2.1** (集合) 集合是一个明确定义的对象集合，记为 $A = \{x \mid P(x)\}$，其中 $P(x)$ 是谓词。

**定义 1.2.2** (集合运算) 对于集合 $A, B$：

- 并集：$A \cup B = \{x \mid x \in A \lor x \in B\}$
- 交集：$A \cap B = \{x \mid x \in A \land x \in B\}$
- 差集：$A \setminus B = \{x \mid x \in A \land x \notin B\}$
- 补集：$A^c = \{x \mid x \notin A\}$

**定理 1.2.1** (德摩根定律) 对于任意集合 $A, B$：
$$(A \cup B)^c = A^c \cap B^c$$
$$(A \cap B)^c = A^c \cup B^c$$

### 关系与函数

**定义 1.2.3** (二元关系) 集合 $A$ 和 $B$ 的二元关系是 $A \times B$ 的子集。

**定义 1.2.4** (函数) 函数 $f: A \rightarrow B$ 是满足以下条件的二元关系：
$$\forall a \in A, \exists! b \in B: (a,b) \in f$$

**定义 1.2.5** (等价关系) 关系 $R \subseteq A \times A$ 是等价关系，如果满足：

1. 自反性：$\forall a \in A: (a,a) \in R$
2. 对称性：$\forall a,b \in A: (a,b) \in R \Rightarrow (b,a) \in R$
3. 传递性：$\forall a,b,c \in A: (a,b) \in R \land (b,c) \in R \Rightarrow (a,c) \in R$

## 1.2.2 图论基础

### 图的基本概念

**定义 1.2.6** (图) 图是一个二元组 $G = (V, E)$，其中：

- $V$ 是顶点集合，满足 $|V| < \infty$
- $E$ 是边集合，满足 $E \subseteq V \times V$

**定义 1.2.7** (有向图) 有向图是一个二元组 $D = (V, A)$，其中：

- $V$ 是顶点集合
- $A$ 是弧集合，满足 $A \subseteq V \times V$

**定义 1.2.8** (路径) 图中的路径是顶点序列 $v_0, v_1, \ldots, v_k$，满足 $(v_i, v_{i+1}) \in E$。

**定理 1.2.2** (最短路径存在性) 在连通图中，任意两个顶点间存在最短路径。

### 图的算法

**算法 1.2.1** (Dijkstra算法) 计算单源最短路径：

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

#[derive(Debug, Clone, Eq, PartialEq)]
struct State {
    cost: i32,
    position: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra(graph: &Vec<Vec<(usize, i32)>>, start: usize) -> Vec<i32> {
    let mut dist = vec![i32::MAX; graph.len()];
    let mut heap = BinaryHeap::new();
    
    dist[start] = 0;
    heap.push(State { cost: 0, position: start });
    
    while let Some(State { cost, position }) = heap.pop() {
        if cost > dist[position] {
            continue;
        }
        
        for &(next, weight) in &graph[position] {
            let next_cost = cost + weight;
            if next_cost < dist[next] {
                dist[next] = next_cost;
                heap.push(State { cost: next_cost, position: next });
            }
        }
    }
    
    dist
}
```

**算法 1.2.2** (Floyd-Warshall算法) 计算全源最短路径：

```rust
fn floyd_warshall(graph: &mut Vec<Vec<i32>>) {
    let n = graph.len();
    
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if graph[i][k] != i32::MAX && graph[k][j] != i32::MAX {
                    graph[i][j] = graph[i][j].min(graph[i][k] + graph[k][j]);
                }
            }
        }
    }
}
```

## 1.2.3 线性代数基础

### 向量空间

**定义 1.2.9** (向量空间) 向量空间 $V$ 是满足以下公理的集合：

1. 加法封闭性：$\forall u,v \in V: u + v \in V$
2. 标量乘法封闭性：$\forall \alpha \in \mathbb{R}, \forall v \in V: \alpha v \in V$
3. 加法交换律：$\forall u,v \in V: u + v = v + u$
4. 加法结合律：$\forall u,v,w \in V: (u + v) + w = u + (v + w)$
5. 零向量存在性：$\exists 0 \in V: \forall v \in V: v + 0 = v$
6. 逆向量存在性：$\forall v \in V, \exists (-v) \in V: v + (-v) = 0$

**定义 1.2.10** (线性无关) 向量组 $\{v_1, v_2, \ldots, v_n\}$ 线性无关，如果：
$$\sum_{i=1}^{n} \alpha_i v_i = 0 \Rightarrow \alpha_i = 0, \forall i$$

**定义 1.2.11** (基) 向量空间 $V$ 的基是线性无关的生成集。

**定理 1.2.3** (基的存在性) 任意有限维向量空间都有基。

### 矩阵理论

**定义 1.2.12** (矩阵) $m \times n$ 矩阵是 $A = [a_{ij}]$，其中 $a_{ij} \in \mathbb{R}$。

**定义 1.2.13** (矩阵乘法) 对于矩阵 $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$：
$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**定义 1.2.14** (特征值) 矩阵 $A$ 的特征值 $\lambda$ 满足：
$$Av = \lambda v$$
其中 $v \neq 0$ 是特征向量。

**定理 1.2.4** (特征值分解) 对于对称矩阵 $A$，存在正交矩阵 $Q$ 和对角矩阵 $\Lambda$：
$$A = Q \Lambda Q^T$$

```rust
use nalgebra::{DMatrix, DVector};

fn eigenvalue_decomposition(matrix: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    // 使用QR算法计算特征值分解
    let (eigenvalues, eigenvectors) = matrix.symmetric_eigen();
    (eigenvectors, eigenvalues.eigenvalues)
}
```

## 1.2.4 概率论基础

### 概率空间

**定义 1.2.15** (概率空间) 概率空间是三元组 $(\Omega, \mathcal{F}, P)$，其中：

- $\Omega$ 是样本空间
- $\mathcal{F}$ 是事件集合，满足 $\sigma$-代数性质
- $P: \mathcal{F} \rightarrow [0,1]$ 是概率测度

**定义 1.2.16** (随机变量) 随机变量 $X: \Omega \rightarrow \mathbb{R}$ 是可测函数。

**定义 1.2.17** (期望) 随机变量 $X$ 的期望：
$$E[X] = \int_{\Omega} X(\omega) dP(\omega)$$

**定义 1.2.18** (方差) 随机变量 $X$ 的方差：
$$\text{Var}(X) = E[(X - E[X])^2]$$

### 概率分布

**定义 1.2.19** (正态分布) $X \sim \mathcal{N}(\mu, \sigma^2)$ 的概率密度函数：
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**定义 1.2.20** (指数分布) $X \sim \text{Exp}(\lambda)$ 的概率密度函数：
$$f(x) = \lambda e^{-\lambda x}, x \geq 0$$

**定理 1.2.5** (中心极限定理) 对于独立同分布的随机变量 $X_1, X_2, \ldots, X_n$：
$$\frac{\sum_{i=1}^{n} X_i - n\mu}{\sqrt{n}\sigma} \xrightarrow{d} \mathcal{N}(0,1)$$

```rust
use rand::distributions::{Normal, Distribution};
use rand::thread_rng;

fn generate_normal_samples(mean: f64, std_dev: f64, n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(mean, std_dev).unwrap();
    
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}
```

## 1.2.5 优化理论

### 凸优化

**定义 1.2.21** (凸集) 集合 $C$ 是凸集，如果：
$$\forall x,y \in C, \forall \lambda \in [0,1]: \lambda x + (1-\lambda)y \in C$$

**定义 1.2.22** (凸函数) 函数 $f: C \rightarrow \mathbb{R}$ 是凸函数，如果：
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**定理 1.2.6** (凸优化最优性) 对于凸优化问题：
$$\min_{x \in C} f(x)$$
如果 $f$ 在 $x^*$ 处可微，则 $x^*$ 是最优解当且仅当：
$$\nabla f(x^*) \cdot (x - x^*) \geq 0, \forall x \in C$$

### 线性规划

**定义 1.2.23** (线性规划) 标准形式线性规划：
$$\begin{align}
\min & \quad c^T x \\
\text{s.t.} & \quad Ax = b \\
& \quad x \geq 0
\end{align}$$

**定理 1.2.7** (对偶性) 原问题和对偶问题的最优值相等。

```rust
use good_lp::{constraint, default_solver, variable, ProblemVariables, Solution, SolverModel};

fn solve_linear_program() -> Result<f64, Box<dyn std::error::Error>> {
    let mut problem = ProblemVariables::new();
    let x1 = problem.add(variable().min(0.0));
    let x2 = problem.add(variable().min(0.0));

    let solution = problem
        .maximise(3.0 * x1 + 2.0 * x2)
        .using(default_solver)
        .with(constraint!(x1 + x2 <= 4.0))
        .with(constraint!(2.0 * x1 + x2 <= 5.0))
        .solve()?;

    Ok(solution.eval(3.0 * x1 + 2.0 * x2))
}
```

## 1.2.6 数值分析

### 数值积分

**定义 1.2.24** (数值积分) 使用数值方法近似计算积分：
$$\int_a^b f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

**算法 1.2.3** (梯形法则)：
$$\int_a^b f(x) dx \approx \frac{h}{2}[f(a) + 2\sum_{i=1}^{n-1} f(x_i) + f(b)]$$

```rust
fn trapezoidal_rule<F>(f: F, a: f64, b: f64, n: usize) -> f64
where F: Fn(f64) -> f64
{
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;

    for i in 1..n {
        sum += f(a + i as f64 * h);
    }

    h * sum
}
```

### 数值微分

**定义 1.2.25** (数值微分) 使用有限差分近似导数：
$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

**算法 1.2.4** (中心差分)：
$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

```rust
fn central_difference<F>(f: F, x: f64, h: f64) -> f64
where F: Fn(f64) -> f64
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}
```

## 1.2.7 离散数学

### 组合数学

**定义 1.2.26** (排列) $n$ 个元素的排列数：
$$P(n,r) = \frac{n!}{(n-r)!}$$

**定义 1.2.27** (组合) $n$ 个元素的组合数：
$$C(n,r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**定理 1.2.8** (二项式定理)：
$$(x+y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$$

### 数论基础

**定义 1.2.28** (最大公约数) $a$ 和 $b$ 的最大公约数 $\gcd(a,b)$ 是最大的正整数 $d$，使得 $d \mid a$ 且 $d \mid b$。

**算法 1.2.5** (欧几里得算法)：

```rust
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
```

**定理 1.2.9** (欧拉定理) 对于互质的整数 $a$ 和 $n$：
$$a^{\phi(n)} \equiv 1 \pmod{n}$$
其中 $\phi(n)$ 是欧拉函数。

## 1.2.8 国际标准对标

### 数学标准

- **ISO 80000-2**: 数学符号和表达式标准
- **IEEE 754**: 浮点数算术标准
- **ISO/IEC 14882**: C++编程语言标准（数学库）
- **ISO/IEC 9899**: C编程语言标准（数学库）

### 学术标准

- **ACM Computing Classification System**: 计算科学分类
- **Mathematics Subject Classification**: 数学主题分类
- **Zentralblatt MATH**: 数学文献数据库标准
- **MathSciNet**: 数学评论数据库标准

## 1.2.9 相关链接

- [1.1 形式化基础理论](./README.md)
- [1.3 语义模型理论](./semantic-models.md)
- [1.4 量子项目管理理论](./quantum-project-theory.md)
- [1.5 生物启发式项目管理理论](./bio-inspired-project-theory.md)
- [1.6 全息项目管理理论](./holographic-project-theory.md)
- [1.7 星际项目管理理论](./interstellar-project-theory.md)
- [2.1 项目生命周期模型](../02-project-management/lifecycle-models.md)
- [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)

## 参考文献

1. Rosen, K. H. (2018). Discrete mathematics and its applications (8th ed.). McGraw-Hill Education.
2. Strang, G. (2016). Introduction to linear algebra (5th ed.). Wellesley-Cambridge Press.
3. Ross, S. M. (2014). A first course in probability (9th ed.). Pearson.
4. Boyd, S., & Vandenberghe, L. (2004). Convex optimization. Cambridge University Press.
5. Burden, R. L., & Faires, J. D. (2010). Numerical analysis (9th ed.). Cengage Learning.
6. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to algorithms (3rd ed.). MIT Press.
7. ISO 80000-2:2019. Quantities and units - Part 2: Mathematics. International Organization for Standardization.
8. IEEE Std 754-2019. IEEE standard for floating-point arithmetic. IEEE Computer Society.
9. ISO/IEC 14882:2020. Programming languages - C++. International Organization for Standardization.
10. ISO/IEC 9899:2018. Programming languages - C. International Organization for Standardization.
