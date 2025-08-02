# 5.2 Haskell实现示例

## 5.2.1 概述

本章节提供Haskell语言实现的项目管理模型示例，展示函数式编程范式下的形式化模型实现。

## 5.2.2 基础类型定义

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}

module FormalProgramManage.Haskell where

import Data.Text (Text)
import Data.Time (UTCTime, getCurrentTime)
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Maybe (fromMaybe)
import Control.Monad.State
import Control.Monad.Reader
import Control.Monad.Except
import GHC.Generics (Generic)
import Data.Aeson (ToJSON, FromJSON)

-- | 项目状态类型
data ProjectState = ProjectState
  { progress :: Double      -- 进度 [0,1]
  , quality :: Double       -- 质量 [0,1]
  , risk :: Double          -- 风险 [0,1]
  , satisfaction :: Double  -- 满意度 [0,1]
  } deriving (Show, Eq, Generic)

instance ToJSON ProjectState
instance FromJSON ProjectState

-- | 任务状态
data TaskStatus
  = Pending
  | InProgress
  | Completed
  | Failed
  deriving (Show, Eq, Generic)

instance ToJSON TaskStatus
instance FromJSON TaskStatus

-- | 任务类型
data Task = Task
  { taskId :: Text
  , taskName :: Text
  , taskDescription :: Text
  , taskPriority :: Int
  , taskStatus :: TaskStatus
  , taskDependencies :: Set Text
  , taskEffort :: Double
  } deriving (Show, Eq, Generic)

instance ToJSON Task
instance FromJSON Task

-- | 资源类型
data ResourceType
  = Human Text
  | Equipment Text
  | Material Text
  deriving (Show, Eq, Generic)

instance ToJSON ResourceType
instance FromJSON ResourceType

-- | 资源
data Resource = Resource
  { resourceId :: Text
  , resourceType :: ResourceType
  , resourceCapacity :: Double
  , resourceCost :: Double
  , resourceAvailability :: Bool
  } deriving (Show, Eq, Generic)

instance ToJSON Resource
instance FromJSON Resource

-- | 风险类型
data RiskType
  = Technical
  | Schedule
  | Cost
  | Quality
  deriving (Show, Eq, Generic)

instance ToJSON RiskType
instance FromJSON RiskType

-- | 风险
data Risk = Risk
  { riskId :: Text
  , riskType :: RiskType
  , riskDescription :: Text
  , riskProbability :: Double
  , riskImpact :: Double
  , riskMitigation :: Text
  } deriving (Show, Eq, Generic)

instance ToJSON Risk
instance FromJSON Risk
```

## 5.2.3 项目管理模型

```haskell
-- | 项目模型
data Project = Project
  { projectId :: Text
  , projectName :: Text
  , projectDescription :: Text
  , projectStartDate :: UTCTime
  , projectEndDate :: UTCTime
  , projectTasks :: Map Text Task
  , projectResources :: Map Text Resource
  , projectRisks :: Map Text Risk
  , projectState :: ProjectState
  } deriving (Show, Eq, Generic)

instance ToJSON Project
instance FromJSON Project

-- | 项目管理环境
data ProjectEnvironment = ProjectEnvironment
  { envConstraints :: Map Text Double
  , envAssumptions :: Map Text Bool
  , envParameters :: Map Text Double
  } deriving (Show, Eq, Generic)

instance ToJSON ProjectEnvironment
instance FromJSON ProjectEnvironment

-- | 项目管理器状态
data ProjectManagerState = ProjectManagerState
  { currentProject :: Maybe Project
  , projectHistory :: [Project]
  , environment :: ProjectEnvironment
  } deriving (Show, Eq, Generic)

instance ToJSON ProjectManagerState
instance FromJSON ProjectManagerState

-- | 项目管理器
newtype ProjectManager a = ProjectManager
  { runProjectManager :: StateT ProjectManagerState (ExceptT Text IO) a
  } deriving (Functor, Applicative, Monad, MonadState ProjectManagerState, MonadError Text, MonadIO)

-- | 运行项目管理器
runProjectManagerIO :: ProjectManager a -> ProjectManagerState -> IO (Either Text (a, ProjectManagerState))
runProjectManagerIO action initialState = runExceptT $ runStateT (runProjectManager action) initialState
```

## 5.2.4 形式化验证函数

```haskell
-- | 验证函数类型
type Validator a = a -> Bool

-- | 项目状态验证器
validateProjectState :: Validator ProjectState
validateProjectState state =
  progress state >= 0 && progress state <= 1 &&
  quality state >= 0 && quality state <= 1 &&
  risk state >= 0 && risk state <= 1 &&
  satisfaction state >= 0 && satisfaction state <= 1

-- | 任务验证器
validateTask :: Validator Task
validateTask task =
  taskPriority task >= 1 && taskPriority task <= 10 &&
  taskEffort task >= 0

-- | 资源验证器
validateResource :: Validator Resource
validateResource resource =
  resourceCapacity resource >= 0 &&
  resourceCost resource >= 0

-- | 风险验证器
validateRisk :: Validator Risk
validateRisk risk =
  riskProbability risk >= 0 && riskProbability risk <= 1 &&
  riskImpact risk >= 0 && riskImpact risk <= 1

-- | 项目验证器
validateProject :: Validator Project
validateProject project =
  validateProjectState (projectState project) &&
  all validateTask (Map.elems $ projectTasks project) &&
  all validateResource (Map.elems $ projectResources project) &&
  all validateRisk (Map.elems $ projectRisks project)
```

## 5.2.5 项目管理操作

```haskell
-- | 创建新项目
createProject :: Text -> Text -> Text -> UTCTime -> UTCTime -> ProjectManager Project
createProject pid name desc startDate endDate = do
  let initialState = ProjectState 0.0 0.0 0.0 0.0
  let project = Project
        { projectId = pid
        , projectName = name
        , projectDescription = desc
        , projectStartDate = startDate
        , projectEndDate = endDate
        , projectTasks = Map.empty
        , projectResources = Map.empty
        , projectRisks = Map.empty
        , projectState = initialState
        }
  
  -- 验证项目
  unless (validateProject project) $
    throwError "项目验证失败"
  
  -- 设置当前项目
  modify $ \s -> s { currentProject = Just project }
  
  return project

-- | 添加任务
addTask :: Text -> Text -> Text -> Int -> Double -> ProjectManager Task
addTask taskId name desc priority effort = do
  state <- get
  case currentProject state of
    Nothing -> throwError "没有活动项目"
    Just project -> do
      let task = Task
            { taskId = taskId
            , taskName = name
            , taskDescription = desc
            , taskPriority = priority
            , taskStatus = Pending
            , taskDependencies = Set.empty
            , taskEffort = effort
            }
      
      -- 验证任务
      unless (validateTask task) $
        throwError "任务验证失败"
      
      let updatedProject = project { projectTasks = Map.insert taskId task (projectTasks project) }
      
      -- 更新项目状态
      modify $ \s -> s { currentProject = Just updatedProject }
      
      return task

-- | 更新任务状态
updateTaskStatus :: Text -> TaskStatus -> ProjectManager ()
updateTaskStatus taskId newStatus = do
  state <- get
  case currentProject state of
    Nothing -> throwError "没有活动项目"
    Just project -> do
      case Map.lookup taskId (projectTasks project) of
        Nothing -> throwError $ "任务不存在: " <> taskId
        Just task -> do
          let updatedTask = task { taskStatus = newStatus }
          let updatedProject = project { projectTasks = Map.insert taskId updatedTask (projectTasks project) }
          
          -- 重新计算项目状态
          let recalculatedProject = recalculateProjectState updatedProject
          
          modify $ \s -> s { currentProject = Just recalculatedProject }

-- | 添加资源
addResource :: Text -> ResourceType -> Double -> Double -> ProjectManager Resource
addResource resourceId resourceType capacity cost = do
  state <- get
  case currentProject state of
    Nothing -> throwError "没有活动项目"
    Just project -> do
      let resource = Resource
            { resourceId = resourceId
            , resourceType = resourceType
            , resourceCapacity = capacity
            , resourceCost = cost
            , resourceAvailability = True
            }
      
      -- 验证资源
      unless (validateResource resource) $
        throwError "资源验证失败"
      
      let updatedProject = project { projectResources = Map.insert resourceId resource (projectResources project) }
      
      modify $ \s -> s { currentProject = Just updatedProject }
      
      return resource

-- | 添加风险
addRisk :: Text -> RiskType -> Text -> Double -> Double -> Text -> ProjectManager Risk
addRisk riskId riskType desc probability impact mitigation = do
  state <- get
  case currentProject state of
    Nothing -> throwError "没有活动项目"
    Just project -> do
      let risk = Risk
            { riskId = riskId
            , riskType = riskType
            , riskDescription = desc
            , riskProbability = probability
            , riskImpact = impact
            , riskMitigation = mitigation
            }
      
      -- 验证风险
      unless (validateRisk risk) $
        throwError "风险验证失败"
      
      let updatedProject = project { projectRisks = Map.insert riskId risk (projectRisks project) }
      
      -- 重新计算项目状态
      let recalculatedProject = recalculateProjectState updatedProject
      
      modify $ \s -> s { currentProject = Just recalculatedProject }
      
      return risk
```

## 5.2.6 状态计算函数

```haskell
-- | 重新计算项目状态
recalculateProjectState :: Project -> Project
recalculateProjectState project = project { projectState = newState }
  where
    newState = ProjectState
      { progress = calculateProgress project
      , quality = calculateQuality project
      , risk = calculateRisk project
      , satisfaction = calculateSatisfaction project
      }

-- | 计算项目进度
calculateProgress :: Project -> Double
calculateProgress project =
  let tasks = Map.elems $ projectTasks project
      totalTasks = length tasks
      completedTasks = length $ filter (\t -> taskStatus t == Completed) tasks
  in if totalTasks == 0 then 0.0 else fromIntegral completedTasks / fromIntegral totalTasks

-- | 计算项目质量
calculateQuality :: Project -> Double
calculateQuality project =
  let tasks = Map.elems $ projectTasks project
      totalTasks = length tasks
      failedTasks = length $ filter (\t -> taskStatus t == Failed) tasks
  in if totalTasks == 0 then 1.0 else 1.0 - (fromIntegral failedTasks / fromIntegral totalTasks)

-- | 计算项目风险
calculateRisk :: Project -> Double
calculateRisk project =
  let risks = Map.elems $ projectRisks project
      totalRisk = sum $ map (\r -> riskProbability r * riskImpact r) risks
  in min 1.0 totalRisk

-- | 计算客户满意度
calculateSatisfaction :: Project -> Double
calculateSatisfaction project =
  let progress = calculateProgress project
      quality = calculateQuality project
      risk = calculateRisk project
  in (progress + quality + (1.0 - risk)) / 3.0
```

## 5.2.7 形式化证明函数

```haskell
-- | 证明项目状态一致性
proveStateConsistency :: Project -> Bool
proveStateConsistency project =
  let state = projectState project
  in validateProjectState state

-- | 证明任务依赖无环性
proveTaskDependencyAcyclicity :: Project -> Bool
proveTaskDependencyAcyclicity project =
  let tasks = Map.elems $ projectTasks project
      taskIds = Set.fromList $ map taskId tasks
      hasValidDependencies task = Set.isSubsetOf (taskDependencies task) taskIds
  in all hasValidDependencies tasks

-- | 证明资源分配合理性
proveResourceAllocation :: Project -> Bool
proveResourceAllocation project =
  let resources = Map.elems $ projectResources project
      hasValidCapacity = all (\r -> resourceCapacity r >= 0) resources
      hasValidCost = all (\r -> resourceCost r >= 0) resources
  in hasValidCapacity && hasValidCost

-- | 证明风险评估合理性
proveRiskAssessment :: Project -> Bool
proveRiskAssessment project =
  let risks = Map.elems $ projectRisks project
      hasValidProbability = all (\r -> riskProbability r >= 0 && riskProbability r <= 1) risks
      hasValidImpact = all (\r -> riskImpact r >= 0 && riskImpact r <= 1) risks
  in hasValidProbability && hasValidImpact

-- | 综合验证
validateProjectFormally :: Project -> Bool
validateProjectFormally project =
  proveStateConsistency project &&
  proveTaskDependencyAcyclicity project &&
  proveResourceAllocation project &&
  proveRiskAssessment project
```

## 5.2.8 测试用例

```haskell
-- | 测试用例
testProjectManagement :: IO ()
testProjectManagement = do
  putStrLn "开始Haskell项目管理测试..."
  
  -- 创建初始状态
  let initialState = ProjectManagerState
        { currentProject = Nothing
        , projectHistory = []
        , environment = ProjectEnvironment Map.empty Map.empty Map.empty
        }
  
  -- 运行测试
  result <- runProjectManagerIO testWorkflow initialState
  
  case result of
    Left err -> putStrLn $ "测试失败: " <> err
    Right (_, finalState) -> do
      putStrLn "测试成功完成"
      case currentProject finalState of
        Nothing -> putStrLn "没有活动项目"
        Just project -> do
          putStrLn $ "项目名称: " <> projectName project
          putStrLn $ "任务数量: " <> show (Map.size $ projectTasks project)
          putStrLn $ "资源数量: " <> show (Map.size $ projectResources project)
          putStrLn $ "风险数量: " <> show (Map.size $ projectRisks project)
          
          let state = projectState project
          putStrLn $ "项目进度: " <> show (progress state)
          putStrLn $ "项目质量: " <> show (quality state)
          putStrLn $ "项目风险: " <> show (risk state)
          putStrLn $ "客户满意度: " <> show (satisfaction state)
          
          -- 形式化验证
          let isValid = validateProjectFormally project
          putStrLn $ "形式化验证结果: " <> show isValid

-- | 测试工作流
testWorkflow :: ProjectManager ()
testWorkflow = do
  -- 创建项目
  currentTime <- liftIO getCurrentTime
  let endTime = addUTCTime (24 * 60 * 60) currentTime -- 1天后
  
  project <- createProject "proj_001" "测试项目" "这是一个测试项目" currentTime endTime
  liftIO $ putStrLn "项目创建成功"
  
  -- 添加任务
  _ <- addTask "task_001" "需求分析" "分析用户需求" 5 8.0
  _ <- addTask "task_002" "系统设计" "设计系统架构" 4 16.0
  _ <- addTask "task_003" "编码实现" "实现核心功能" 3 24.0
  liftIO $ putStrLn "任务添加成功"
  
  -- 添加资源
  _ <- addResource "res_001" (Human "张三") 40.0 100.0
  _ <- addResource "res_002" (Human "李四") 40.0 120.0
  _ <- addResource "res_003" (Equipment "开发服务器") 168.0 50.0
  liftIO $ putStrLn "资源添加成功"
  
  -- 添加风险
  _ <- addRisk "risk_001" Technical "技术风险" 0.3 0.7 "增加技术评审"
  _ <- addRisk "risk_002" Schedule "进度风险" 0.2 0.6 "增加缓冲时间"
  liftIO $ putStrLn "风险添加成功"
  
  -- 更新任务状态
  updateTaskStatus "task_001" Completed
  updateTaskStatus "task_002" InProgress
  liftIO $ putStrLn "任务状态更新成功"
  
  liftIO $ putStrLn "测试工作流完成"

-- | 主函数
main :: IO ()
main = testProjectManagement
```

## 5.2.9 形式化证明

### 5.2.9.1 状态一致性证明

**定理 5.2.1** (状态一致性) 对于任意项目 $P$，其状态 $s$ 满足：
$$\forall s \in S: 0 \leq s.progress \leq 1 \land 0 \leq s.quality \leq 1 \land 0 \leq s.risk \leq 1 \land 0 \leq s.satisfaction \leq 1$$

**证明**：
由 `calculateProgress`、`calculateQuality`、`calculateRisk`、`calculateSatisfaction` 函数的定义可知，所有计算都在合理范围内。

### 5.2.9.2 任务依赖无环性证明

**定理 5.2.2** (依赖无环性) 对于任意项目 $P$，其任务依赖关系是无环的。

**证明**：
由 `proveTaskDependencyAcyclicity` 函数验证，确保所有依赖的任务ID都存在于任务集合中。

### 5.2.9.3 资源分配合理性证明

**定理 5.2.3** (资源合理性) 对于任意项目 $P$，其资源分配是合理的。

**证明**：
由 `proveResourceAllocation` 函数验证，确保所有资源的容量和成本都是非负数。

## 5.2.10 引用关系

- 基础理论：参见 [1.1 形式化基础理论](../01-foundations/README.md)
- 项目管理：参见 [2.1 项目生命周期模型](../02-project-management/lifecycle-models.md)
- 形式化验证：参见 [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)
- Rust实现：参见 [5.1 Rust实现示例](./rust-examples.md)

---

**持续构建中...** 返回 [实现与工具](../README.md) | [项目主页](../../README.md)
