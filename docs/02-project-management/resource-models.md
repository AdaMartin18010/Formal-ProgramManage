# 2.2 资源管理模型

## 概述

资源管理模型是Formal-ProgramManage的核心理论之一，定义了项目资源的优化配置、分配和监控机制。本理论体系严格对标PMBOK 7th Edition、ISO 21500、PRINCE2等国际项目管理标准。

## 2.2.1 资源管理基础理论

### 基本定义

**定义 2.2.1** (项目资源 - PMBOK 7th Edition) 项目资源是一个四元组：
$$\mathcal{R} = (H, M, T, F)$$

其中：

- $H = \{h_1, h_2, \ldots, h_n\}$ 是人力资源集合，满足 $h_i \in \mathbb{R}^+$
- $M = \{m_1, m_2, \ldots, m_k\}$ 是物质资源集合，满足 $m_i \in \mathbb{R}^+$
- $T = \{t_1, t_2, \ldots, t_l\}$ 是技术资源集合，满足 $t_i \in \mathbb{R}^+$
- $F = \{f_1, f_2, \ldots, f_m\}$ 是财务资源集合，满足 $f_i \in \mathbb{R}^+$

**定义 2.2.2** (资源分配函数) 资源分配函数是一个映射：
$$\text{allocate}: \mathcal{T} \times \mathcal{R} \rightarrow \mathbb{R}^+$$

其中 $\mathcal{T}$ 是任务集合，满足：
$$\forall t \in \mathcal{T}, \forall r \in \mathcal{R}: \text{allocate}(t, r) \geq 0$$

**定义 2.2.3** (资源约束) 资源约束是一个三元组：
$$C = (R, L, U)$$

其中：

- $R$ 是资源类型
- $L$ 是下界约束，满足 $L \in \mathbb{R}^+$
- $U$ 是上界约束，满足 $U \in \mathbb{R}^+$ 且 $U \geq L$

## 2.2.2 资源优化模型

### 线性规划模型

**定义 2.2.4** (资源优化问题) 资源优化问题是一个线性规划：
$$\begin{align}
\text{minimize} \quad & \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} \\
\text{subject to} \quad & \sum_{j=1}^{m} x_{ij} \leq a_i, \quad i = 1, 2, \ldots, n \\
& \sum_{i=1}^{n} x_{ij} \geq b_j, \quad j = 1, 2, \ldots, m \\
& x_{ij} \geq 0, \quad \forall i, j
\end{align}$$

其中：
- $x_{ij}$ 是分配给任务 $i$ 的资源 $j$ 的数量
- $c_{ij}$ 是单位成本
- $a_i$ 是资源 $i$ 的可用量
- $b_j$ 是任务 $j$ 的需求量

### 动态规划模型

**定义 2.2.5** (资源动态规划) 资源动态规划的状态转移方程：
$$V(i, r) = \max_{0 \leq x \leq r} \{v_i(x) + V(i-1, r-x)\}$$

其中：
- $V(i, r)$ 是前 $i$ 个任务使用 $r$ 单位资源的最大价值
- $v_i(x)$ 是任务 $i$ 使用 $x$ 单位资源的价值
- $x$ 是分配给任务 $i$ 的资源量

## 2.2.3 资源调度算法

### 关键路径法

**算法 2.2.1** (关键路径资源调度)：

```rust
use std::collections::{HashMap, HashSet, VecDeque};

# [derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub duration: f64,
    pub resource_requirements: HashMap<String, f64>,
    pub dependencies: Vec<String>,
    pub earliest_start: f64,
    pub latest_start: f64,
    pub slack: f64,
}

# [derive(Debug, Clone)]
pub struct Resource {
    pub id: String,
    pub capacity: f64,
    pub cost_per_unit: f64,
    pub availability: Vec<(f64, f64)>, // (start_time, end_time)
}

# [derive(Debug)]
pub struct ResourceScheduler {
    pub tasks: HashMap<String, Task>,
    pub resources: HashMap<String, Resource>,
    pub schedule: HashMap<String, Vec<(f64, f64, f64)>>, // task_id -> [(start, end, resource_amount)]
}

impl ResourceScheduler {
    pub fn new() -> Self {
        ResourceScheduler {
            tasks: HashMap::new(),
            resources: HashMap::new(),
            schedule: HashMap::new(),
        }
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.insert(task.id.clone(), task);
    }

    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.id.clone(), resource);
    }

    pub fn calculate_critical_path(&self) -> Vec<String> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut earliest_start: HashMap<String, f64> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        // 初始化入度
        for task_id in self.tasks.keys() {
            in_degree.insert(task_id.clone(), 0);
        }

        // 计算入度
        for task in self.tasks.values() {
            for dep in &task.dependencies {
                *in_degree.get_mut(dep).unwrap() += 1;
            }
        }

        // 拓扑排序
        for (task_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(task_id.clone());
                earliest_start.insert(task_id.clone(), 0.0);
            }
        }

        let mut critical_path = Vec::new();

        while let Some(task_id) = queue.pop_front() {
            let task = &self.tasks[&task_id];
            let current_earliest = earliest_start[&task_id];

            // 更新后续任务的最早开始时间
            for (next_id, next_task) in &self.tasks {
                if next_task.dependencies.contains(&task_id) {
                    let new_earliest = current_earliest + task.duration;
                    let current = earliest_start.get(next_id).unwrap_or(&0.0);
                    earliest_start.insert(next_id.clone(), new_earliest.max(*current));

                    *in_degree.get_mut(next_id).unwrap() -= 1;
                    if in_degree[next_id] == 0 {
                        queue.push_back(next_id.clone());
                    }
                }
            }

            critical_path.push(task_id);
        }

        critical_path
    }

    pub fn optimize_resource_allocation(&mut self) -> f64 {
        let mut total_cost = 0.0;

        // 按关键路径顺序分配资源
        let critical_path = self.calculate_critical_path();

        for task_id in critical_path {
            let task = &self.tasks[&task_id];
            let mut best_allocation = HashMap::new();
            let mut min_cost = f64::INFINITY;

            // 尝试不同的资源分配方案
            for resource_id in task.resource_requirements.keys() {
                let resource = &self.resources[resource_id];
                let required = task.resource_requirements[resource_id];

                // 计算最优分配
                let optimal_amount = self.calculate_optimal_allocation(
                    task, resource_id, required
                );

                best_allocation.insert(resource_id.clone(), optimal_amount);
                min_cost += optimal_amount * resource.cost_per_unit;
            }

            // 更新调度
            self.schedule.insert(task_id.clone(), vec![
                (task.earliest_start, task.earliest_start + task.duration, min_cost)
            ]);

            total_cost += min_cost;
        }

        total_cost
    }

    fn calculate_optimal_allocation(&self, task: &Task, resource_id: &str, required: f64) -> f64 {
        let resource = &self.resources[resource_id];

        // 考虑资源可用性和成本
        let available = self.get_available_resource(resource_id, task.earliest_start, task.earliest_start + task.duration);
        let optimal = required.min(available);

        optimal
    }

    fn get_available_resource(&self, resource_id: &str, start_time: f64, end_time: f64) -> f64 {
        let resource = &self.resources[resource_id];

        // 检查时间窗口内的可用性
        let mut available = resource.capacity;

        for (avail_start, avail_end) in &resource.availability {
            if start_time >= *avail_start && end_time <= *avail_end {
                available = available.min(resource.capacity);
            }
        }

        available
    }

    pub fn calculate_resource_utilization(&self) -> HashMap<String, f64> {
        let mut utilization = HashMap::new();

        for (resource_id, resource) in &self.resources {
            let mut total_used = 0.0;
            let mut total_available = 0.0;

            for (task_id, allocations) in &self.schedule {
                for (start, end, amount) in allocations {
                    let duration = end - start;
                    total_used += amount * duration;
                }
            }

            // 计算总可用时间
            for (start, end) in &resource.availability {
                total_available += resource.capacity * (end - start);
            }

            let util = if total_available > 0.0 {
                total_used / total_available
            } else {
                0.0
            };

            utilization.insert(resource_id.clone(), util);
        }

        utilization
    }
}
```

### 遗传算法优化

**算法 2.2.2** (遗传算法资源优化)：

```rust
use std::collections::HashMap;
use rand::Rng;

# [derive(Debug, Clone)]
pub struct Chromosome {
    pub gene: Vec<f64>, // 资源分配方案
    pub fitness: f64,
}

# [derive(Debug)]
pub struct GeneticOptimizer {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub generations: usize,
    pub tasks: Vec<Task>,
    pub resources: Vec<Resource>,
}

impl GeneticOptimizer {
    pub fn new(population_size: usize, tasks: Vec<Task>, resources: Vec<Resource>) -> Self {
        GeneticOptimizer {
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            generations: 100,
            tasks,
            resources,
        }
    }

    pub fn optimize(&mut self) -> Chromosome {
        let mut population = self.initialize_population();

        for generation in 0..self.generations {
            // 计算适应度
            for chromosome in &mut population {
                chromosome.fitness = self.calculate_fitness(&chromosome.gene);
            }

            // 排序
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

            // 选择、交叉、变异
            let mut new_population = Vec::new();

            // 精英保留
            let elite_size = self.population_size / 10;
            for i in 0..elite_size {
                new_population.push(population[i].clone());
            }

            // 生成新个体
            while new_population.len() < self.population_size {
                let parent1 = self.tournament_selection(&population);
                let parent2 = self.tournament_selection(&population);

                let (child1, child2) = self.crossover(&parent1, &parent2);

                let child1 = self.mutate(child1);
                let child2 = self.mutate(child2);

                new_population.push(child1);
                if new_population.len() < self.population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;

            if generation % 10 == 0 {
                println!("Generation {}: Best fitness = {}", generation, population[0].fitness);
            }
        }

        population[0].clone()
    }

    fn initialize_population(&self) -> Vec<Chromosome> {
        let mut rng = rand::thread_rng();
        let mut population = Vec::new();

        for _ in 0..self.population_size {
            let mut gene = Vec::new();

            for task in &self.tasks {
                for resource in &self.resources {
                    let allocation = rng.gen_range(0.0..resource.capacity);
                    gene.push(allocation);
                }
            }

            population.push(Chromosome {
                gene,
                fitness: 0.0,
            });
        }

        population
    }

    fn calculate_fitness(&self, gene: &[f64]) -> f64 {
        let mut total_cost = 0.0;
        let mut constraint_violation = 0.0;

        let mut gene_index = 0;

        for task in &self.tasks {
            for resource in &self.resources {
                let allocation = gene[gene_index];

                // 计算成本
                total_cost += allocation * resource.cost_per_unit;

                // 检查约束违反
                if allocation > resource.capacity {
                    constraint_violation += allocation - resource.capacity;
                }

                gene_index += 1;
            }
        }

        // 适应度 = 1 / (成本 + 惩罚项)
        1.0 / (total_cost + 1000.0 * constraint_violation)
    }

    fn tournament_selection(&self, population: &[Chromosome]) -> &Chromosome {
        let mut rng = rand::thread_rng();
        let tournament_size = 3;

        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.fitness > best.fitness {
                best = candidate;
            }
        }

        best
    }

    fn crossover(&self, parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() > self.crossover_rate {
            return (parent1.clone(), parent2.clone());
        }

        let crossover_point = rng.gen_range(0..parent1.gene.len());

        let mut child1_gene = parent1.gene.clone();
        let mut child2_gene = parent2.gene.clone();

        for i in crossover_point..parent1.gene.len() {
            child1_gene[i] = parent2.gene[i];
            child2_gene[i] = parent1.gene[i];
        }

        (Chromosome { gene: child1_gene, fitness: 0.0 },
         Chromosome { gene: child2_gene, fitness: 0.0 })
    }

    fn mutate(&self, mut chromosome: Chromosome) -> Chromosome {
        let mut rng = rand::thread_rng();

        for i in 0..chromosome.gene.len() {
            if rng.gen::<f64>() < self.mutation_rate {
                let resource_index = i % self.resources.len();
                let resource = &self.resources[resource_index];
                chromosome.gene[i] = rng.gen_range(0.0..resource.capacity);
            }
        }

        chromosome
    }
}
```

## 2.2.4 资源监控与控制

### 资源监控系统

**定义 2.2.6** (资源监控指标) 资源监控指标包括：
- **资源利用率**: $\text{Utilization} = \frac{\text{Used}}{\text{Available}} \times 100\%$
- **资源效率**: $\text{Efficiency} = \frac{\text{Output}}{\text{Input}}$
- **资源成本**: $\text{Cost} = \sum_{i} c_i \times r_i$
- **资源可用性**: $\text{Availability} = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$

### 资源控制算法

**算法 2.2.3** (资源控制算法)：

```rust
use std::collections::HashMap;

# [derive(Debug)]
pub struct ResourceController {
    pub target_utilization: f64,
    pub control_threshold: f64,
    pub adjustment_rate: f64,
    pub historical_data: Vec<ResourceMetrics>,
}

# [derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub timestamp: f64,
    pub utilization: f64,
    pub efficiency: f64,
    pub cost: f64,
    pub availability: f64,
}

impl ResourceController {
    pub fn new(target_utilization: f64) -> Self {
        ResourceController {
            target_utilization,
            control_threshold: 0.1,
            adjustment_rate: 0.05,
            historical_data: Vec::new(),
        }
    }

    pub fn monitor_resources(&mut self, current_metrics: ResourceMetrics) -> Vec<ResourceAdjustment> {
        self.historical_data.push(current_metrics.clone());

        let mut adjustments = Vec::new();

        // 检查利用率偏差
        let utilization_deviation = (current_metrics.utilization - self.target_utilization).abs();

        if utilization_deviation > self.control_threshold {
            let adjustment = self.calculate_adjustment(&current_metrics);
            adjustments.push(adjustment);
        }

        // 检查效率趋势
        if self.historical_data.len() >= 3 {
            let efficiency_trend = self.calculate_efficiency_trend();
            if efficiency_trend < 0.0 {
                let efficiency_adjustment = self.calculate_efficiency_adjustment();
                adjustments.push(efficiency_adjustment);
            }
        }

        adjustments
    }

    fn calculate_adjustment(&self, metrics: &ResourceMetrics) -> ResourceAdjustment {
        let deviation = metrics.utilization - self.target_utilization;
        let adjustment_amount = deviation * self.adjustment_rate;

        ResourceAdjustment {
            resource_id: "general".to_string(),
            adjustment_type: if deviation > 0.0 {
                AdjustmentType::Reduce
            } else {
                AdjustmentType::Increase
            },
            amount: adjustment_amount.abs(),
            reason: format!("Utilization deviation: {:.2}%", deviation * 100.0),
        }
    }

    fn calculate_efficiency_trend(&self) -> f64 {
        let n = self.historical_data.len();
        let recent_efficiency: f64 = self.historical_data[n-3..].iter()
            .map(|m| m.efficiency)
            .sum::<f64>() / 3.0;

        let previous_efficiency: f64 = self.historical_data[n-6..n-3].iter()
            .map(|m| m.efficiency)
            .sum::<f64>() / 3.0;

        recent_efficiency - previous_efficiency
    }

    fn calculate_efficiency_adjustment(&self) -> ResourceAdjustment {
        ResourceAdjustment {
            resource_id: "efficiency".to_string(),
            adjustment_type: AdjustmentType::Optimize,
            amount: 0.1,
            reason: "Declining efficiency trend detected".to_string(),
        }
    }
}

# [derive(Debug)]
pub struct ResourceAdjustment {
    pub resource_id: String,
    pub adjustment_type: AdjustmentType,
    pub amount: f64,
    pub reason: String,
}

# [derive(Debug)]
pub enum AdjustmentType {
    Increase,
    Reduce,
    Optimize,
    Reallocate,
}
```

## 2.2.5 国际标准对标

### PMBOK 7th Edition 标准

- **资源管理知识领域**: 项目资源管理过程
- **资源规划**: 规划资源管理、估算活动资源
- **资源获取**: 获取资源、建设团队、管理团队
- **资源控制**: 控制资源

### ISO 21500 标准

- **资源管理过程**: 资源管理相关过程
- **资源分配**: 资源分配和优化
- **资源监控**: 资源使用监控和控制

### PRINCE2 标准

- **资源主题**: 资源管理主题
- **资源计划**: 资源计划和分配
- **资源控制**: 资源使用控制

## 2.2.6 相关链接

- [2.1 项目生命周期模型](./lifecycle-models.md)
- [2.3 风险管理模型](./risk-models.md)
- [2.4 质量管理模型](./quality-models.md)
- [1.1 形式化基础理论](../01-foundations/README.md)
- [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)

## 参考文献

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
3. AXELOS. (2017). Managing Successful Projects with PRINCE2 2017 Edition. TSO (The Stationery Office).
4. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling (12th ed.). John Wiley & Sons.
5. Meredith, J. R., & Mantel, S. J. (2019). Project management: a managerial approach (10th ed.). John Wiley & Sons.
6. Turner, J. R. (2016). Gower handbook of project management (5th ed.). Routledge.
7. Lock, D. (2013). Project management (10th ed.). Routledge.
8. Schwalbe, K. (2019). Information technology project management (9th ed.). Cengage Learning.
9. Wysocki, R. K. (2019). Effective project management: traditional, agile, extreme, hybrid (8th ed.). John Wiley & Sons.
10. Goldratt, E. M. (1997). Critical chain. North River Press.
