# 1.2 数学模型基础

## 概述

数学模型基础为Formal-ProgramManage提供严格的数学工具和理论框架，确保项目管理的精确性和可验证性。

## 1.2.1 集合论基础

### 项目集合定义

**定义 1.2.1** 项目集合 $\mathcal{P}$ 是所有可能项目的幂集：
$$\mathcal{P} = 2^{\mathcal{U}}$$

其中 $\mathcal{U}$ 是项目宇宙。

### 关系理论

**定义 1.2.2** 项目依赖关系 $D \subseteq \mathcal{P} \times \mathcal{P}$ 满足：

- 自反性：$\forall p \in \mathcal{P}: (p,p) \in D$
- 传递性：$(p_1,p_2) \in D \land (p_2,p_3) \in D \Rightarrow (p_1,p_3) \in D$

## 1.2.2 图论模型

### 项目依赖图

**定义 1.2.3** 项目依赖图 $G = (V, E)$ 是一个有向无环图，其中：

- $V$ 是项目节点集合
- $E \subseteq V \times V$ 是依赖边集合

### 拓扑排序

**算法 1.2.1** 项目拓扑排序算法：

```rust
use std::collections::{HashMap, VecDeque};

pub struct ProjectGraph {
    pub nodes: Vec<String>,
    pub edges: HashMap<String, Vec<String>>,
    pub in_degree: HashMap<String, usize>,
}

impl ProjectGraph {
    pub fn topological_sort(&self) -> Option<Vec<String>> {
        let mut in_degree = self.in_degree.clone();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        // 找到所有入度为0的节点
        for node in &self.nodes {
            if in_degree.get(node).unwrap_or(&0) == &0 {
                queue.push_back(node.clone());
            }
        }
        
        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            
            // 更新邻居节点的入度
            if let Some(neighbors) = self.edges.get(&node) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }
        
        if result.len() == self.nodes.len() {
            Some(result)
        } else {
            None // 存在环
        }
    }
}
```

## 1.2.3 线性代数模型

### 资源分配矩阵

**定义 1.2.4** 资源分配矩阵 $A \in \mathbb{R}^{n \times m}$，其中：

- $n$ 是项目数量
- $m$ 是资源类型数量
- $A_{ij}$ 表示项目 $i$ 对资源 $j$ 的需求量

### 约束条件

**定义 1.2.5** 资源约束满足：
$$A \cdot x \leq b$$

其中：

- $x \in \mathbb{R}^n$ 是项目执行向量
- $b \in \mathbb{R}^m$ 是资源容量向量

## 1.2.4 概率论模型

### 项目风险模型

**定义 1.2.6** 项目风险函数 $R: \mathcal{P} \rightarrow [0,1]$：
$$R(p) = \sum_{i=1}^{n} w_i \cdot P(failure_i)$$

其中：

- $w_i$ 是风险权重
- $P(failure_i)$ 是第 $i$ 个失败事件的概率

### 蒙特卡洛模拟

**算法 1.2.2** 项目风险蒙特卡洛模拟：

```rust
use rand::Rng;

pub struct RiskSimulation {
    pub iterations: usize,
    pub risk_factors: Vec<f64>,
    pub weights: Vec<f64>,
}

impl RiskSimulation {
    pub fn simulate_risk(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let mut total_risk = 0.0;
        
        for _ in 0..self.iterations {
            let mut iteration_risk = 0.0;
            for (factor, weight) in self.risk_factors.iter().zip(self.weights.iter()) {
                if rng.gen::<f64>() < *factor {
                    iteration_risk += weight;
                }
            }
            total_risk += iteration_risk;
        }
        
        total_risk / self.iterations as f64
    }
}
```

## 1.2.5 优化理论

### 线性规划模型

**定义 1.2.7** 项目优化问题：
$$\min_{x} c^T x$$
$$\text{subject to } Ax \leq b, x \geq 0$$

其中：

- $c \in \mathbb{R}^n$ 是成本向量
- $x \in \mathbb{R}^n$ 是决策变量

### 动态规划

**定义 1.2.8** 项目价值函数递推：
$$V_t(s) = \max_{a \in A(s)} \left\{ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_{t+1}(s') \right\}$$

## 1.2.6 微分方程模型

### 项目进度微分方程

**定义 1.2.9** 项目进度微分方程：
$$\frac{dP}{dt} = f(P, R, t)$$

其中：

- $P(t)$ 是时间 $t$ 的项目进度
- $R(t)$ 是时间 $t$ 的资源投入
- $f$ 是进度函数

### 解析解

**定理 1.2.1** 如果 $f(P, R, t) = k \cdot R(t) \cdot (1 - P(t))$，则：
$$P(t) = 1 - (1 - P_0) \cdot e^{-k \int_0^t R(\tau) d\tau}$$

## 1.2.7 信息论模型

### 项目信息熵

**定义 1.2.10** 项目信息熵：
$$H(P) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

其中 $p_i$ 是第 $i$ 个状态的概率。

### 互信息

**定义 1.2.11** 项目间互信息：
$$I(P_1; P_2) = H(P_1) + H(P_2) - H(P_1, P_2)$$

## 1.2.8 实现示例

### Haskell 实现

```haskell
-- 项目依赖图
data ProjectGraph = ProjectGraph {
    nodes :: [String],
    edges :: [(String, String)]
}

-- 拓扑排序
topologicalSort :: ProjectGraph -> Maybe [String]
topologicalSort graph = 
    let inDegree = foldl (\acc (from, to) -> 
            Map.insertWith (+) to 1 acc) Map.empty (edges graph)
        queue = [node | node <- nodes graph, 
                       Map.findWithDefault 0 node inDegree == 0]
    in sortHelper graph inDegree queue []

sortHelper :: ProjectGraph -> Map String Int -> [String] -> [String] -> Maybe [String]
sortHelper _ _ [] result = Just result
sortHelper graph inDegree (node:queue) result =
    let newInDegree = foldl (\acc neighbor -> 
            Map.insertWith (+) neighbor (-1) acc) inDegree 
            [to | (from, to) <- edges graph, from == node]
        newQueue = queue ++ [n | n <- nodes graph, 
                                Map.findWithDefault 0 n newInDegree == 0,
                                n `notElem` (node:queue)]
    in sortHelper graph newInDegree newQueue (result ++ [node])
```

## 1.2.9 相关链接

- [1.1 形式化基础理论](./README.md)
- [1.3 语义模型理论](./semantic-models.md)
- [2.2 资源管理模型](../02-project-management/resource-models.md)
- [2.3 风险管理模型](../02-project-management/risk-models.md)

## 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to algorithms. MIT press.
2. Boyd, S., & Vandenberghe, L. (2004). Convex optimization. Cambridge university press.
3. Cover, T. M., & Thomas, J. A. (2012). Elements of information theory. John Wiley & Sons.
