# 2.2 资源管理模型

## 概述

资源管理模型是Formal-ProgramManage的核心组成部分，提供严格的数学框架来管理项目中的各种资源，包括人力资源、物质资源、财务资源和时间资源。

## 2.2.1 资源基础定义

### 资源类型定义

**定义 2.2.1** 项目资源是一个四元组 $R = (T, C, A, U)$，其中：

- $T$ 是资源类型 (Resource Type)
- $C$ 是容量 (Capacity)
- $A$ 是可用性 (Availability)
- $U$ 是使用率 (Utilization)

### 资源类型分类

**定义 2.2.2** 资源类型集合 $\mathcal{T} = \{Human, Equipment, Material, Financial, Time\}$

**定义 2.2.3** 人力资源 $R_{human} = (Human, C_{human}, A_{human}, U_{human})$

**定义 2.2.4** 设备资源 $R_{equipment} = (Equipment, C_{equipment}, A_{equipment}, U_{equipment})$

**定义 2.2.5** 物质资源 $R_{material} = (Material, C_{material}, A_{material}, U_{material})$

**定义 2.2.6** 财务资源 $R_{financial} = (Financial, C_{financial}, A_{financial}, U_{financial})$

**定义 2.2.7** 时间资源 $R_{time} = (Time, C_{time}, A_{time}, U_{time})$

## 2.2.2 资源分配理论

### 资源分配函数

**定义 2.2.8** 资源分配函数 $RA: \mathcal{R} \times \mathcal{T} \times \mathcal{P} \rightarrow \mathbb{R}^+$ 满足：
$$\forall r \in \mathcal{R}, \forall t \in \mathcal{T}, \forall p \in \mathcal{P}: RA(r, t, p) \geq 0$$

### 资源约束条件

**定义 2.2.9** 容量约束：
$$\forall r \in \mathcal{R}, \forall t \in \mathcal{T}: \sum_{p \in \mathcal{P}} RA(r, t, p) \leq C(r)$$

**定义 2.2.10** 可用性约束：
$$\forall r \in \mathcal{R}, \forall t \in \mathcal{T}: RA(r, t, p) \leq A(r, t)$$

### 资源分配优化

**定义 2.2.11** 资源分配优化问题：
$$\min_{RA} \sum_{r \in \mathcal{R}} \sum_{t \in \mathcal{T}} \sum_{p \in \mathcal{P}} c(r, t, p) \cdot RA(r, t, p)$$

约束条件：
$$\sum_{p \in \mathcal{P}} RA(r, t, p) \leq C(r), \quad \forall r \in \mathcal{R}, \forall t \in \mathcal{T}$$
$$\sum_{r \in \mathcal{R}} RA(r, t, p) \geq D(p, t), \quad \forall p \in \mathcal{P}, \forall t \in \mathcal{T}$$

其中：

- $c(r, t, p)$ 是资源成本函数
- $D(p, t)$ 是项目需求函数

## 2.2.3 资源调度模型

### 调度问题定义

**定义 2.2.12** 资源调度问题是一个五元组 $SP = (J, R, P, C, O)$，其中：

- $J$ 是任务集合
- $R$ 是资源集合
- $P$ 是优先级关系
- $C$ 是约束条件
- $O$ 是目标函数

### 调度算法

**算法 2.2.1** 最早截止时间优先调度 (EDF)：

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub duration: u64,
    pub deadline: u64,
    pub priority: u32,
    pub resource_requirements: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Resource {
    pub id: String,
    pub capacity: f64,
    pub cost_per_unit: f64,
    pub availability: Vec<TimeSlot>,
}

#[derive(Debug, Clone)]
pub struct TimeSlot {
    pub start_time: u64,
    pub end_time: u64,
    pub available_capacity: f64,
}

#[derive(Debug, Clone)]
pub struct Schedule {
    pub task_assignments: HashMap<String, TaskAssignment>,
    pub resource_utilization: HashMap<String, Vec<TimeSlot>>,
    pub total_cost: f64,
}

#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub task_id: String,
    pub resource_id: String,
    pub start_time: u64,
    pub end_time: u64,
    pub allocated_capacity: f64,
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for Task {}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        // EDF: 按截止时间排序
        self.deadline.cmp(&other.deadline)
    }
}

pub struct ResourceScheduler {
    pub tasks: Vec<Task>,
    pub resources: HashMap<String, Resource>,
    pub schedule: Schedule,
}

impl ResourceScheduler {
    pub fn new() -> Self {
        ResourceScheduler {
            tasks: Vec::new(),
            resources: HashMap::new(),
            schedule: Schedule {
                task_assignments: HashMap::new(),
                resource_utilization: HashMap::new(),
                total_cost: 0.0,
            },
        }
    }
    
    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }
    
    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.id.clone(), resource);
    }
    
    pub fn schedule_edf(&mut self) -> Result<Schedule, String> {
        // 按截止时间排序任务
        let mut sorted_tasks = self.tasks.clone();
        sorted_tasks.sort(); // 使用Ord trait排序
        
        let mut schedule = Schedule {
            task_assignments: HashMap::new(),
            resource_utilization: HashMap::new(),
            total_cost: 0.0,
        };
        
        for task in sorted_tasks {
            let assignment = self.find_best_assignment(&task)?;
            schedule.task_assignments.insert(task.id.clone(), assignment.clone());
            
            // 更新资源利用率
            self.update_resource_utilization(&mut schedule, &assignment);
            
            // 计算成本
            let resource = self.resources.get(&assignment.resource_id)
                .ok_or("Resource not found")?;
            let cost = assignment.allocated_capacity * resource.cost_per_unit * 
                      (assignment.end_time - assignment.start_time) as f64;
            schedule.total_cost += cost;
        }
        
        self.schedule = schedule.clone();
        Ok(schedule)
    }
    
    fn find_best_assignment(&self, task: &Task) -> Result<TaskAssignment, String> {
        let mut best_assignment = None;
        let mut best_cost = f64::INFINITY;
        
        for (resource_id, resource) in &self.resources {
            // 检查资源是否满足任务需求
            if self.can_allocate_resource(resource_id, task) {
                let assignment = self.create_assignment(task, resource_id)?;
                let cost = self.calculate_assignment_cost(&assignment);
                
                if cost < best_cost {
                    best_cost = cost;
                    best_assignment = Some(assignment);
                }
            }
        }
        
        best_assignment.ok_or("No suitable resource found".to_string())
    }
    
    fn can_allocate_resource(&self, resource_id: &str, task: &Task) -> bool {
        let resource = self.resources.get(resource_id).unwrap();
        
        // 检查资源需求
        for (req_resource_id, req_capacity) in &task.resource_requirements {
            if req_resource_id == resource_id && req_capacity > &resource.capacity {
                return false;
            }
        }
        
        true
    }
    
    fn create_assignment(&self, task: &Task, resource_id: &str) -> Result<TaskAssignment, String> {
        let resource = self.resources.get(resource_id)
            .ok_or("Resource not found")?;
        
        // 找到最早的可用时间槽
        let start_time = self.find_earliest_available_slot(resource_id, task)?;
        let end_time = start_time + task.duration;
        
        let allocated_capacity = task.resource_requirements.get(resource_id)
            .unwrap_or(&0.0);
        
        Ok(TaskAssignment {
            task_id: task.id.clone(),
            resource_id: resource_id.to_string(),
            start_time,
            end_time,
            allocated_capacity: *allocated_capacity,
        })
    }
    
    fn find_earliest_available_slot(&self, resource_id: &str, task: &Task) -> Result<u64, String> {
        // 简化实现：返回当前时间
        Ok(0)
    }
    
    fn calculate_assignment_cost(&self, assignment: &TaskAssignment) -> f64 {
        let resource = self.resources.get(&assignment.resource_id).unwrap();
        assignment.allocated_capacity * resource.cost_per_unit * 
        (assignment.end_time - assignment.start_time) as f64
    }
    
    fn update_resource_utilization(&self, schedule: &mut Schedule, assignment: &TaskAssignment) {
        let resource_util = schedule.resource_utilization
            .entry(assignment.resource_id.clone())
            .or_insert_with(Vec::new);
        
        resource_util.push(TimeSlot {
            start_time: assignment.start_time,
            end_time: assignment.end_time,
            available_capacity: assignment.allocated_capacity,
        });
    }
}
```

## 2.2.4 资源优化理论

### 线性规划模型

**定义 2.2.13** 资源优化线性规划模型：
$$\min_{x} c^T x$$

约束条件：
$$Ax \leq b$$
$$x \geq 0$$

其中：

- $x$ 是决策变量向量
- $c$ 是成本向量
- $A$ 是约束矩阵
- $b$ 是约束向量

### 动态规划优化

**定义 2.2.14** 资源优化动态规划：
$$V_t(s) = \min_{a \in A(s)} \{c(s, a) + V_{t+1}(f(s, a))\}$$

其中：

- $V_t(s)$ 是时间 $t$ 状态 $s$ 的最优值函数
- $c(s, a)$ 是状态 $s$ 下动作 $a$ 的成本
- $f(s, a)$ 是状态转移函数

### 遗传算法优化

**算法 2.2.2** 资源优化遗传算法：

```rust
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Chromosome {
    pub genes: Vec<u32>,
    pub fitness: f64,
}

#[derive(Debug, Clone)]
pub struct GeneticOptimizer {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub generations: usize,
}

impl GeneticOptimizer {
    pub fn new(population_size: usize, mutation_rate: f64, crossover_rate: f64, generations: usize) -> Self {
        GeneticOptimizer {
            population_size,
            mutation_rate,
            crossover_rate,
            generations,
        }
    }
    
    pub fn optimize(&self, tasks: &[Task], resources: &HashMap<String, Resource>) -> Vec<TaskAssignment> {
        let mut population = self.initialize_population(tasks, resources);
        
        for generation in 0..self.generations {
            // 评估适应度
            for chromosome in &mut population {
                chromosome.fitness = self.calculate_fitness(chromosome, tasks, resources);
            }
            
            // 选择
            let selected = self.selection(&population);
            
            // 交叉
            let offspring = self.crossover(&selected);
            
            // 变异
            let mutated = self.mutation(&offspring);
            
            // 更新种群
            population = mutated;
        }
        
        // 返回最优解
        let best_chromosome = population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();
        
        self.decode_chromosome(best_chromosome, tasks, resources)
    }
    
    fn initialize_population(&self, tasks: &[Task], resources: &HashMap<String, Resource>) -> Vec<Chromosome> {
        let mut population = Vec::new();
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.population_size {
            let mut genes = Vec::new();
            for _ in 0..tasks.len() {
                genes.push(rng.gen_range(0..resources.len() as u32));
            }
            
            population.push(Chromosome {
                genes,
                fitness: 0.0,
            });
        }
        
        population
    }
    
    fn calculate_fitness(&self, chromosome: &Chromosome, tasks: &[Task], resources: &HashMap<String, Resource>) -> f64 {
        let assignments = self.decode_chromosome(chromosome, tasks, resources);
        
        // 计算总成本
        let total_cost = assignments.iter()
            .map(|assignment| {
                let resource = resources.get(&assignment.resource_id).unwrap();
                assignment.allocated_capacity * resource.cost_per_unit * 
                (assignment.end_time - assignment.start_time) as f64
            })
            .sum::<f64>();
        
        // 计算约束违反惩罚
        let penalty = self.calculate_constraint_penalty(&assignments, resources);
        
        // 适应度 = 1 / (成本 + 惩罚)
        1.0 / (total_cost + penalty)
    }
    
    fn calculate_constraint_penalty(&self, assignments: &[TaskAssignment], resources: &HashMap<String, Resource>) -> f64 {
        let mut penalty = 0.0;
        
        for (resource_id, resource) in resources {
            let total_allocation = assignments.iter()
                .filter(|a| a.resource_id == *resource_id)
                .map(|a| a.allocated_capacity)
                .sum::<f64>();
            
            if total_allocation > resource.capacity {
                penalty += (total_allocation - resource.capacity) * 1000.0;
            }
        }
        
        penalty
    }
    
    fn selection(&self, population: &[Chromosome]) -> Vec<Chromosome> {
        // 轮盘赌选择
        let total_fitness: f64 = population.iter().map(|c| c.fitness).sum();
        let mut selected = Vec::new();
        let mut rng = rand::thread_rng();
        
        for _ in 0..population.len() {
            let random = rng.gen_range(0.0..total_fitness);
            let mut cumulative = 0.0;
            
            for chromosome in population {
                cumulative += chromosome.fitness;
                if cumulative >= random {
                    selected.push(chromosome.clone());
                    break;
                }
            }
        }
        
        selected
    }
    
    fn crossover(&self, parents: &[Chromosome]) -> Vec<Chromosome> {
        let mut offspring = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..parents.len() - 1 {
            if rng.gen::<f64>() < self.crossover_rate {
                let parent1 = &parents[i];
                let parent2 = &parents[i + 1];
                
                let (child1, child2) = self.single_point_crossover(parent1, parent2);
                offspring.push(child1);
                offspring.push(child2);
            } else {
                offspring.push(parents[i].clone());
                offspring.push(parents[i + 1].clone());
            }
        }
        
        offspring
    }
    
    fn single_point_crossover(&self, parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0..parent1.genes.len());
        
        let mut child1_genes = parent1.genes.clone();
        let mut child2_genes = parent2.genes.clone();
        
        for i in crossover_point..parent1.genes.len() {
            child1_genes[i] = parent2.genes[i];
            child2_genes[i] = parent1.genes[i];
        }
        
        (Chromosome { genes: child1_genes, fitness: 0.0 },
         Chromosome { genes: child2_genes, fitness: 0.0 })
    }
    
    fn mutation(&self, population: &[Chromosome]) -> Vec<Chromosome> {
        let mut mutated = Vec::new();
        let mut rng = rand::thread_rng();
        
        for chromosome in population {
            let mut new_genes = chromosome.genes.clone();
            
            for i in 0..new_genes.len() {
                if rng.gen::<f64>() < self.mutation_rate {
                    new_genes[i] = rng.gen_range(0..new_genes.len() as u32);
                }
            }
            
            mutated.push(Chromosome {
                genes: new_genes,
                fitness: 0.0,
            });
        }
        
        mutated
    }
    
    fn decode_chromosome(&self, chromosome: &Chromosome, tasks: &[Task], resources: &HashMap<String, Resource>) -> Vec<TaskAssignment> {
        let mut assignments = Vec::new();
        let resource_ids: Vec<String> = resources.keys().cloned().collect();
        
        for (i, task) in tasks.iter().enumerate() {
            let resource_index = chromosome.genes[i] as usize;
            let resource_id = resource_ids[resource_index].clone();
            
            assignments.push(TaskAssignment {
                task_id: task.id.clone(),
                resource_id,
                start_time: 0, // 简化实现
                end_time: task.duration,
                allocated_capacity: task.resource_requirements.get(&resource_id).unwrap_or(&0.0),
            });
        }
        
        assignments
    }
}
```

## 2.2.5 资源监控模型

### 资源利用率监控

**定义 2.2.15** 资源利用率函数：
$$U(r, t) = \frac{\sum_{p \in \mathcal{P}} RA(r, t, p)}{C(r)}$$

### 资源效率指标

**定义 2.2.16** 资源效率指标：
$$E(r) = \frac{\sum_{t \in \mathcal{T}} U(r, t) \cdot V(r, t)}{\sum_{t \in \mathcal{T}} V(r, t)}$$

其中 $V(r, t)$ 是资源价值函数。

### 监控实现

```rust
pub struct ResourceMonitor {
    pub utilization_history: HashMap<String, Vec<UtilizationRecord>>,
    pub efficiency_metrics: HashMap<String, f64>,
    pub alerts: Vec<ResourceAlert>,
}

#[derive(Debug, Clone)]
pub struct UtilizationRecord {
    pub timestamp: u64,
    pub utilization: f64,
    pub allocated: f64,
    pub available: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceAlert {
    pub resource_id: String,
    pub alert_type: AlertType,
    pub severity: Severity,
    pub message: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    OverUtilization,
    UnderUtilization,
    ResourceConflict,
    CapacityExceeded,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        ResourceMonitor {
            utilization_history: HashMap::new(),
            efficiency_metrics: HashMap::new(),
            alerts: Vec::new(),
        }
    }
    
    pub fn record_utilization(&mut self, resource_id: &str, utilization: f64, allocated: f64, available: f64) {
        let record = UtilizationRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            utilization,
            allocated,
            available,
        };
        
        self.utilization_history
            .entry(resource_id.to_string())
            .or_insert_with(Vec::new)
            .push(record);
        
        // 检查告警条件
        self.check_alerts(resource_id, utilization);
    }
    
    fn check_alerts(&mut self, resource_id: &str, utilization: f64) {
        if utilization > 0.9 {
            self.alerts.push(ResourceAlert {
                resource_id: resource_id.to_string(),
                alert_type: AlertType::OverUtilization,
                severity: Severity::High,
                message: format!("Resource {} is over-utilized: {:.2}%", resource_id, utilization * 100.0),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        } else if utilization < 0.1 {
            self.alerts.push(ResourceAlert {
                resource_id: resource_id.to_string(),
                alert_type: AlertType::UnderUtilization,
                severity: Severity::Medium,
                message: format!("Resource {} is under-utilized: {:.2}%", resource_id, utilization * 100.0),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        }
    }
    
    pub fn calculate_efficiency(&mut self, resource_id: &str) -> f64 {
        if let Some(history) = self.utilization_history.get(resource_id) {
            let total_utilization: f64 = history.iter().map(|r| r.utilization).sum();
            let average_utilization = total_utilization / history.len() as f64;
            
            self.efficiency_metrics.insert(resource_id.to_string(), average_utilization);
            average_utilization
        } else {
            0.0
        }
    }
    
    pub fn get_resource_report(&self, resource_id: &str) -> ResourceReport {
        let history = self.utilization_history.get(resource_id).unwrap_or(&Vec::new());
        let efficiency = self.efficiency_metrics.get(resource_id).unwrap_or(&0.0);
        
        let alerts = self.alerts.iter()
            .filter(|alert| alert.resource_id == resource_id)
            .cloned()
            .collect();
        
        ResourceReport {
            resource_id: resource_id.to_string(),
            utilization_history: history.clone(),
            efficiency: *efficiency,
            alerts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub resource_id: String,
    pub utilization_history: Vec<UtilizationRecord>,
    pub efficiency: f64,
    pub alerts: Vec<ResourceAlert>,
}
```

## 2.2.6 相关链接

- [1.1 形式化基础理论](../01-foundations/README.md)
- [1.2 数学模型基础](../01-foundations/mathematical-models.md)
- [2.1 项目生命周期模型](./lifecycle-models.md)
- [2.3 风险管理模型](./risk-models.md)
- [2.4 质量管理模型](./quality-models.md)

## 参考文献

1. Pinedo, M. (2016). Scheduling: theory, algorithms, and systems. Springer.
2. Goldberg, D. E. (1989). Genetic algorithms in search, optimization and machine learning. Addison-Wesley.
3. Bertsekas, D. P. (2017). Dynamic programming and optimal control. Athena Scientific.
4. Hillier, F. S., & Lieberman, G. J. (2015). Introduction to operations research. McGraw-Hill.
