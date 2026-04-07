# BaozhiRAG 协作规范

本文件只定义本仓库内用户、人类开发者与智能代理之间的协作规则，不再承担项目实现规范职责。

- 项目实现规范、代码分层、技术约束、风控边界、测试与提交流程，以 [`_bmad-output/project-context.md`](./_bmad-output/project-context.md) 为准
- 若 `AGENTS.md` 与 `project-context.md` 在代码实现层面发生冲突，以 `project-context.md` 为准
- 若与法律、合规、安全要求冲突，以更严格要求为准

## 1. 文档分工

- `AGENTS.md`
  - 面向“怎么协作”
  - 约束代理与用户之间的沟通、交付物落点、执行方式
- `_bmad-output/project-context.md`
  - 面向“怎么实现”
  - 约束项目代码、架构、测试、风控、流程与实现边界
- `README.md`
  - 面向“怎么使用项目”
  - 提供启动、调试、部署和功能说明

## 2. 协作原则

- 代理在开始执行前，应先理解用户当前目标，再阅读与目标直接相关的规范和上下文
- 代理优先做小而完整的改动，避免一次引入过多抽象
- 代理修改文档、目录或命令后，要同步更新 README 或相关规范文件
- 用户若明确指定输出路径、产物类型或工作流，代理应优先遵守
- 用户未明确指定时，代理应按仓库现有 BMad 配置和目录职责落盘

## 3. 输出物落点

当前仓库按 BMad 配置解析后的输出位置如下：

- `output_folder` -> `E:\PracticalProject\BaozhiRAG\_bmad-output`
- `planning_artifacts` -> `E:\PracticalProject\BaozhiRAG\_bmad-output\planning-artifacts`
- `implementation_artifacts` -> `E:\PracticalProject\BaozhiRAG\_bmad-output\implementation-artifacts`
- `project_knowledge` -> `E:\PracticalProject\BaozhiRAG\_bmad-output\project-knowledge`

落盘约定如下：

- 规划类文档写入 `planning_artifacts`
- 实现类文档写入 `implementation_artifacts`
- 通用输出写入 `output_folder`
- 项目知识沉淀和 brownfield 扫描文档写入 `project_knowledge`
- 文件命名优先遵循 BMad 工作流模板、模块帮助表和技能说明；若模板已规定文件名、目录层级或状态字段，不额外施加仓库级命名约束

## 4. 用户交互约定

- 用户提出“继续完成”“接着做”这类请求时，代理应先检查当前工作流状态与已有产物，再继续执行，而不是从头开始
- 用户若仅提出方向性目标，代理可结合仓库上下文做合理假设，但应在完成后说明关键假设
- 遇到高风险、破坏性或会影响大量现有工作的决策时，代理应先与用户对齐
- 用户可把 `AGENTS.md` 视为协作约定，把 `project-context.md` 视为项目实现规范

## 5. 维护要求

- 当协作方式、BMad 输出目录或文档职责发生变化时，优先更新本文件
- 当技术栈、架构、测试门禁、风控规则或实现约束发生变化时，更新 `_bmad-output/project-context.md`
- 当启动方式、环境变量、调试步骤或功能说明发生变化时，更新 `README.md`

## 6、Agents

### `analyst`

- 路径：`.agents/skills/bmad-agent-analyst/SKILL.md`
- 说明：战略业务分析与需求梳理专家；当用户想与 Mary 对话或请求业务分析师时使用。

### `architect`

- 路径：`.agents/skills/bmad-agent-architect/SKILL.md`
- 说明：系统架构与技术设计负责人；当用户想与 Winston 对话或请求架构师时使用。

### `builder`

- 路径：`.agents/skills/bmad-agent-builder/SKILL.md`
- 说明：通过对话式探索来构建、编辑或分析 Agent Skill；当用户请求创建、分析或编辑 Agent 时使用。

### `dev`

- 路径：`.agents/skills/bmad-agent-dev/SKILL.md`
- 说明：负责故事执行与代码实现的高级软件工程师；当用户想与 Amelia 对话或请求开发者 agent 时使用。

### `pm`

- 路径：`.agents/skills/bmad-agent-pm/SKILL.md`
- 说明：负责 PRD 编写与需求探索的产品经理；当用户想与 John 对话或请求产品经理时使用。

### `qa`

- 路径：`.agents/skills/bmad-agent-qa/SKILL.md`
- 说明：负责测试自动化与覆盖率的 QA 工程师；当用户想与 Quinn 对话或请求 QA 工程师时使用。

### `quick-flow-solo-dev`

- 路径：`.agents/skills/bmad-agent-quick-flow-solo-dev/SKILL.md`
- 说明：用于快速规格设计与实现的顶级全栈开发者；当用户想与 Barry 对话或请求 quick flow solo dev 时使用。

### `sm`

- 路径：`.agents/skills/bmad-agent-sm/SKILL.md`
- 说明：负责冲刺规划与故事准备的 Scrum Master；当用户想与 Bob 对话或请求 Scrum Master 时使用。

### `tech-writer`

- 路径：`.agents/skills/bmad-agent-tech-writer/SKILL.md`
- 说明：技术文档专家与知识整理者；当用户想与 Paige 对话或请求技术写作者时使用。

### `ux-designer`

- 路径：`.agents/skills/bmad-agent-ux-designer/SKILL.md`
- 说明：UX 设计师与 UI 专家；当用户想与 Sally 对话或请求 UX 设计师时使用。

### `brainstorming-coach`

- 路径：`.agents/skills/bmad-cis-agent-brainstorming-coach/SKILL.md`
- 说明：负责引导式创意发想会议的顶级头脑风暴专家；当用户想与 Carson 对话或请求 Brainstorming Specialist 时使用。

### `creative-problem-solver`

- 路径：`.agents/skills/bmad-cis-agent-creative-problem-solver/SKILL.md`
- 说明：精通系统化问题解决方法的高级问题解决专家；当用户想与 Dr. Quinn 对话或请求 Master Problem Solver 时使用。

### `design-thinking-coach`

- 路径：`.agents/skills/bmad-cis-agent-design-thinking-coach/SKILL.md`
- 说明：专注于以人为本设计流程的设计思维大师；当用户想与 Maya 对话或请求 Design Thinking Maestro 时使用。

### `innovation-strategist`

- 路径：`.agents/skills/bmad-cis-agent-innovation-strategist/SKILL.md`
- 说明：聚焦商业模式创新与战略性颠覆的颠覆式创新专家；当用户想与 Victor 对话或请求 Disruptive Innovation Oracle 时使用。

### `presentation-master`

- 路径：`.agents/skills/bmad-cis-agent-presentation-master/SKILL.md`
- 说明：擅长幻灯片、路演稿与视觉叙事的视觉传达与演示专家；当用户想与 Caravaggio 对话或请求 Presentation Expert 时使用。

### `storyteller`

- 路径：`.agents/skills/bmad-cis-agent-storyteller/SKILL.md`
- 说明：运用成熟框架打造有感染力叙事的故事大师；当用户想与 Sophia 对话或请求 Master Storyteller 时使用。

### `tea`

- 路径：`.agents/skills/bmad-tea/SKILL.md`
- 说明：首席测试架构师与质量顾问；当用户想与 Murat 对话或请求 Test Architect 时使用。
