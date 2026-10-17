# RLax

![CI status](https://github.com/deepmind/rlax/workflows/ci/badge.svg)
![docs](https://readthedocs.org/projects/rlax/badge/?version=latest)
![pypi](https://img.shields.io/pypi/v/rlax)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

RLax（发音为“relax”）是一个基于 JAX 构建的开源库，为实现强化学习（Reinforcement Learning, RL）智能体提供实用的核心构建算子与模块。完整文档请参阅 [rlax.readthedocs.io](https://rlax.readthedocs.io/en/latest/index.html)。

## 安装 (Installation)

你可以通过 PyPI 安装 RLax 的最新正式发布版：

```sh
pip install rlax
```

或者直接从 GitHub 安装最新的开发版本：

```sh
pip install git+https://github.com/deepmind/rlax.git
```

RLax 的所有代码都可以通过 `jax.jit` 针对不同的计算硬件（例如 CPU、GPU、TPU）进行即时编译（JIT, Just-In-Time）。

为了运行 `examples/` 中的示例脚本，你需要克隆本仓库并安装额外的依赖库：[optax](https://github.com/deepmind/optax)、[haiku](https://github.com/deepmind/haiku) 以及 [bsuite](https://github.com/deepmind/bsuite)。

## 核心内容 (Content)

RLax 提供的操作与函数并非端到端的完整算法，而是构建具备自主学习能力的功能完备智能体时所需的专门数学运算算子：

* **价值（Values）**：包含状态价值（State-Value）与动作价值（Action-Value）；
* **非线性贝尔曼方程泛化**：用于贝尔曼方程非线性扩展形式的价值计算；
* **回报分布（Return Distributions）**：即分布强化学习价值函数（Distributional Value Functions）；
* **通用价值函数（General Value Functions, GVF）**：用于主任务奖励之外其他累积量（cumulants）的价值评估；
* **策略（Policies）**：通过策略梯度（Policy Gradients）在连续动作空间与离散动作空间中更新策略。

本库同时支持同策略（on-policy）与异策略（off-policy，即从与智能体当前策略不同的数据分布中采样学习）学习。

关于各函数的详细文档以及提出或使用这些算子的学术论文引用，请参阅具体文件与函数级的 docstring 注释。

## 使用方法 (Usage)

请参阅 `examples/` 目录，了解如何使用 RLax 中的算子实现一些经典的基础强化学习智能体，并在 BSuite 的 Catch 环境（强化学习文献中用于智能体开发的经典基准单元测试）上演示训练效果：

在 [bsuite](https://github.com/deepmind/bsuite/tree/master/bsuite/baselines) 中还可以找到更多使用 `rlax` 构建 JAX 强化学习智能体的基线示例。

## 背景知识 (Background)

强化学习致力于研究学习系统（即**智能体 / agent**）如何学会在所处的交互环境中（即**环境 / environment**）采取行动。

智能体与环境在离散时间步上进行交互。在每一个时间步中，智能体选择一个**动作 (action)**，随后环境返回其当前状态的（局部）快照（即**观测 / observation**）以及一个标量反馈信号（即**奖励 / reward**）。

智能体的行为特征由以过去环境观测为条件的动作概率分布（即**策略 / policy**）决定。智能体的目标是寻找一个最优策略，该策略能从任意时间步开始，最大化后续收集的折现累积奖励（即**回报 / return**）。

通常，智能体策略本身或环境动态是具有随机性的。在这种情况下，回报是一个随机变量，最优智能体策略通常被精确定义为：在智能体与环境的随机性下，最大化回报期望（即**价值 / value**）的策略。

## 强化学习算法体系 (Reinforcement Learning Algorithms)

强化学习算法通常分为三大典型流派：

1. **值函数方法**：估计状态与动作的价值，并通过**评估检验 (inspection)** 推断策略（例如始终选择估计价值最高的动作）；
2. **基于模型的方法**：学习环境的模型（能够预测未来的观测与奖励），并通过**规划 (planning)** 推断策略；
3. **策略搜索方法**：参数化一个可直接**执行 (executed)** 的策略。

无论采用何种形式，策略、价值或环境模型本质上都是函数。在深度强化学习中，这些函数由神经网络表示。在这种场景下，强化学习的更新通常被形式化为可微的伪损失函数（pseudo-loss functions，类似于无监督/监督学习）。借助自动微分技术即可精确还原原始的强化学习参数更新规则。

但需要特别注意的是，这些更新仅在输入数据按符合理论假设的方式采样时才有效。例如，策略梯度损失仅在输入轨迹是从当前策略中无偏采样（即同策略 / on-policy 数据）时才成立。本库在代码层面无法强制检查此类约束。各函数 docstring 中提供了描述具体运算使用方式的相关论文链接供查阅。

## 命名规范与开发指南 (Naming Conventions and Developer Guidelines)

我们面向与单个经验流（single stream of experience）交互的智能体定义函数与算子。可以通过 JAX 的核心机制 `vmap` 将相同的函数无缝应用到批量数据（例如支持经验回放 / replay 与并行环境数据生成）。

许多函数需要根据连续时间步中的策略、动作、奖励与价值来计算其输出。在这种情况下，通常使用后缀 `_t` 和 `_tm1` 来明确每个输入的生成时间步，例如：

* `q_tm1`：转移在源状态（`source` state）下的动作价值；
* `a_tm1`：在源状态下所选择执行的动作；
* `r_t`：在目标状态（`destination` state）收集到的即时奖励；
* `discount_t`：与当前转移关联的折现系数（`discount`）；
* `q_t`：在目标状态下的动作价值。

RLax 为每个函数提供了广泛完备的单元测试。所有测试均需验证：当通过 `jax.jit` 编译为 XLA 原生指令，以及通过 `jax.vmap` 执行批量批处理操作时，`rlax` 函数的输出正确无误。

## 引用 RLax (Citing RLax)

本代码库是 DeepMind JAX 生态系统（DeepMind JAX Ecosystem）的一部分，如果你的研究或项目使用了 RLax，请引用以下文献：

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月18日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
