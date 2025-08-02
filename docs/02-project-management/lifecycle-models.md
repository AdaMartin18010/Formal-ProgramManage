# 2.1 项目生命周期模型

## 概述

项目生命周期模型是Formal-ProgramManage的核心组成部分，定义了项目从启动到结束的完整过程，并提供形式化规范来确保项目管理的系统性和可预测性。

## 2.1.1 基本生命周期定义

### 项目生命周期

**定义 2.1.1** 项目生命周期是一个五元组 $LC = (S, T, \delta, \lambda, F)$，其中：

- $S$ 是生命周期状态集合
- $T$ 是时间域
- $\delta: S \times T \rightarrow S$ 是状态转换函数
- $\lambda: S \rightarrow \mathcal{L}$ 是状态标签函数
- $F \subseteq S$ 是最终状态集合

### 生命周期阶段

**定义 2.1.2** 标准项目生命周期包含以下阶段：

1. **启动阶段** (Initiation): $s_0 \in S_{init}$
2. **规划阶段** (Planning): $s_1 \in S_{plan}$
3. **执行阶段** (Execution): $s_2 \in S_{exec}$
4. **监控阶段** (Monitoring): $s_3 \in S_{monitor}$
5. **收尾阶段** (Closure): $s_4 \in S_{close}$

## 2.1.2 瀑布模型

### 瀑布模型定义

**定义 2.1.3** 瀑布模型是一个严格的线性生命周期模型：
$$Waterfall = (S_{wf}, T_{wf}, \delta_{wf}, \lambda_{wf}, F_{wf})$$

其中状态转换满足：
$$\forall s_i, s_j \in S_{wf}: \delta_{wf}(s_i, t) = s_j \Rightarrow i < j$$

### 瀑布模型状态转换

**算法 2.1.1** 瀑布模型状态转换算法：

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum WaterfallPhase {
    Requirements,
    Design,
    Implementation,
    Verification,
    Maintenance,
}

pub struct WaterfallModel {
    pub current_phase: WaterfallPhase,
    pub deliverables: HashMap<WaterfallPhase, Vec<String>>,
    pub phase_completion: HashMap<WaterfallPhase, f64>,
}

impl WaterfallModel {
    pub fn new() -> Self {
        WaterfallModel {
            current_phase: WaterfallPhase::Requirements,
            deliverables: HashMap::new(),
            phase_completion: HashMap::new(),
        }
    }
    
    pub fn advance_phase(&mut self) -> Result<(), String> {
        match self.current_phase {
            WaterfallPhase::Requirements => {
                if self.is_phase_complete(WaterfallPhase::Requirements) {
                    self.current_phase = WaterfallPhase::Design;
                    Ok(())
                } else {
                    Err("Requirements phase not complete".to_string())
                }
            },
            WaterfallPhase::Design => {
                if self.is_phase_complete(WaterfallPhase::Design) {
                    self.current_phase = WaterfallPhase::Implementation;
                    Ok(())
                } else {
                    Err("Design phase not complete".to_string())
                }
            },
            // ... 其他阶段转换
            _ => Err("Invalid phase transition".to_string())
        }
    }
    
    fn is_phase_complete(&self, phase: WaterfallPhase) -> bool {
        self.phase_completion.get(&phase).unwrap_or(&0.0) >= 1.0
    }
}
```

## 2.1.3 敏捷模型

### 敏捷迭代模型

**定义 2.1.4** 敏捷迭代模型是一个循环生命周期模型：
$$Agile = (S_{agile}, T_{agile}, \delta_{agile}, \lambda_{agile}, F_{agile})$$

其中状态转换满足循环性质：
$$\exists n \in \mathbb{N}: \delta_{agile}^n(s) = s$$

### Sprint 定义

**定义 2.1.5** Sprint 是一个时间盒迭代：
$$Sprint = (T_{sprint}, B_{sprint}, V_{sprint})$$

其中：

- $T_{sprint}$ 是固定时间周期
- $B_{sprint}$ 是待办事项集合
- $V_{sprint}$ 是完成价值

### 敏捷实现

```rust
#[derive(Debug, Clone)]
pub struct Sprint {
    pub id: u32,
    pub duration_days: u32,
    pub backlog: Vec<UserStory>,
    pub completed_stories: Vec<UserStory>,
    pub velocity: f64,
}

#[derive(Debug, Clone)]
pub struct UserStory {
    pub id: String,
    pub title: String,
    pub description: String,
    pub story_points: u32,
    pub priority: u32,
    pub status: StoryStatus,
}

#[derive(Debug, Clone)]
pub enum StoryStatus {
    ToDo,
    InProgress,
    Done,
}

pub struct AgileModel {
    pub sprints: Vec<Sprint>,
    pub current_sprint: Option<u32>,
    pub product_backlog: Vec<UserStory>,
}

impl AgileModel {
    pub fn start_sprint(&mut self, sprint_id: u32) -> Result<(), String> {
        if let Some(sprint) = self.sprints.iter_mut().find(|s| s.id == sprint_id) {
            self.current_sprint = Some(sprint_id);
            Ok(())
        } else {
            Err("Sprint not found".to_string())
        }
    }
    
    pub fn complete_story(&mut self, story_id: &str) -> Result<(), String> {
        if let Some(sprint) = self.current_sprint.and_then(|id| 
            self.sprints.iter_mut().find(|s| s.id == id)) {
            
            if let Some(story_index) = sprint.backlog.iter().position(|s| s.id == story_id) {
                let story = sprint.backlog.remove(story_index);
                sprint.completed_stories.push(story);
                Ok(())
            } else {
                Err("Story not found in current sprint".to_string())
            }
        } else {
            Err("No active sprint".to_string())
        }
    }
}
```

## 2.1.4 螺旋模型

### 螺旋模型定义

**定义 2.1.6** 螺旋模型是一个风险驱动的迭代模型：
$$Spiral = (S_{spiral}, T_{spiral}, \delta_{spiral}, \lambda_{spiral}, F_{spiral})$$

其中每个迭代包含四个象限：

1. **目标确定** (Objective Setting)
2. **风险分析** (Risk Analysis)
3. **开发验证** (Development & Validation)
4. **计划下一迭代** (Planning Next Iteration)

### 风险驱动决策

**定义 2.1.7** 螺旋模型风险函数：
$$R_{spiral}(i) = \sum_{j=1}^{n} w_j \cdot risk_j(i)$$

其中 $i$ 是迭代次数，$risk_j(i)$ 是第 $j$ 个风险在迭代 $i$ 中的评估值。

## 2.1.5 形式化验证

### 生命周期属性

**定义 2.1.8** 生命周期安全性属性：
$$\mathbf{G}(current\_phase \in S_{valid})$$

**定义 2.1.9** 生命周期活性属性：
$$\mathbf{G}\mathbf{F}(phase\_complete)$$

### 状态转换验证

**定理 2.1.1** 生命周期状态转换一致性

**定理** 对于任意生命周期模型 $LC$，如果状态转换函数 $\delta$ 是确定的，则：
$$\forall s \in S, \forall t_1, t_2 \in T: \delta(s, t_1) = \delta(s, t_2) \Rightarrow t_1 = t_2$$

**证明**：

1. 假设存在 $t_1 \neq t_2$ 使得 $\delta(s, t_1) = \delta(s, t_2)$
2. 由于 $\delta$ 是确定的，这导致矛盾
3. 因此 $t_1 = t_2$

## 2.1.6 混合模型

### 混合生命周期

**定义 2.1.10** 混合生命周期模型：
$$Hybrid = \alpha \cdot Waterfall + \beta \cdot Agile + \gamma \cdot Spiral$$

其中 $\alpha + \beta + \gamma = 1$ 是权重系数。

### 自适应模型

**定义 2.1.11** 自适应生命周期模型根据项目特征动态调整：
$$Adaptive(t) = f(Complexity(t), Risk(t), Team(t))$$

## 2.1.7 实现示例

### Lean 实现

```lean
-- 生命周期状态
inductive LifecyclePhase
| Initiation
| Planning  
| Execution
| Monitoring
| Closure

-- 状态转换关系
inductive PhaseTransition : LifecyclePhase → LifecyclePhase → Prop
| init_to_plan : PhaseTransition Initiation Planning
| plan_to_exec : PhaseTransition Planning Execution
| exec_to_monitor : PhaseTransition Execution Monitoring
| monitor_to_closure : PhaseTransition Monitoring Closure

-- 生命周期模型
structure LifecycleModel :=
(phases : List LifecyclePhase)
(transitions : List (LifecyclePhase × LifecyclePhase))
(current_phase : LifecyclePhase)
(completion_status : LifecyclePhase → Float)

-- 验证属性
theorem phase_transition_valid (m : LifecycleModel) :
  ∀ p1 p2 : LifecyclePhase,
  (p1, p2) ∈ m.transitions →
  p1 ∈ m.phases ∧ p2 ∈ m.phases :=
begin
  -- 证明实现
end
```

## 2.1.8 相关链接

- [1.1 形式化基础理论](../01-foundations/README.md)
- [2.2 资源管理模型](./resource-models.md)
- [2.3 风险管理模型](./risk-models.md)
- [2.4 质量管理模型](./quality-models.md)
- [4.1 软件开发模型](../04-industry-applications/software-development.md)

## 参考文献

1. Royce, W. W. (1970). Managing the development of large software systems. IEEE WESCON, 26(8), 1-9.
2. Beck, K., Beedle, M., Van Bennekum, A., Cockburn, A., Cunningham, W., Fowler, M., ... & Sutherland, J. (2001). Manifesto for agile software development.
3. Boehm, B. W. (1988). A spiral model of software development and enhancement. Computer, 21(5), 61-72.
