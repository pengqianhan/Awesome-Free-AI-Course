<div align="center">

# Awesome Free AI Course

**免费的人工智能 / 深度学习学习资料合集，以及我在学习过程中整理的笔记与代码**

*A curated collection of free AI & deep learning learning resources, plus personal study notes and code.*

[![GitHub stars](https://img.shields.io/github/stars/pengqianhan/Awesome-Free-AI-Course?style=flat-square)](https://github.com/pengqianhan/Awesome-Free-AI-Course/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/pengqianhan/Awesome-Free-AI-Course?style=flat-square)](https://github.com/pengqianhan/Awesome-Free-AI-Course/network/members)
[![Last commit](https://img.shields.io/github/last-commit/pengqianhan/Awesome-Free-AI-Course?style=flat-square)](https://github.com/pengqianhan/Awesome-Free-AI-Course/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/pengqianhan/Awesome-Free-AI-Course/pulls)

</div>

---

## 📖 目录

- [项目简介](#-项目简介)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [内容导航](#-内容导航)
  - [学习资料合集](#-学习资料合集-resources)
  - [学习笔记](#-学习笔记-notes)
- [贡献指南](#-贡献指南)
- [致谢与声明](#-致谢与声明)

## 🌟 项目简介

这个仓库有两部分内容：

1. **学习资料合集**：网上收集的经典、免费的 AI 学习资源，覆盖 Python、机器学习、深度学习、强化学习、图神经网络、LLM、Diffusion 等方向，不定期更新。
2. **学习笔记与代码**：我在学习和科研过程中写的笔记、公式推导和带注释的代码，包括 nanoGPT、minimind、VAE / VQ-VAE、生存分析、混合自动机等主题。

适合想系统入门 AI 的同学，也适合想找某个具体主题资料的读者。

## 🚀 快速开始

| 如果你是…… | 推荐从这里开始 |
| --- | --- |
| 零基础入门，或想找成体系的课程和书籍 | [深度学习资料合集](resources/deeplearning_material.md) |
| 想专门学习 Transformer | [Awesome Transformer Learning](resources/Awesome-Transformer-learning/Chinese_version.md)（[English](resources/Awesome-Transformer-learning/English_version.md)） |
| 想从零理解 / 训练一个 LLM | [nanoGPT 学习笔记](notes/LLM学习笔记/nanoGPT学习笔记) · [minimind 学习笔记](notes/LLM学习笔记/minimind学习笔记) |
| 准备写论文 | [论文写作资料](resources/writing_papers.md) |

在本地阅读或运行代码：

```bash
git clone https://github.com/pengqianhan/Awesome-Free-AI-Course.git
cd Awesome-Free-AI-Course
```

> 仓库中的 `.ipynb` 文件可以直接在 GitHub 上预览，也可以用 Jupyter / VS Code 打开运行。

## 🗂 项目结构

```text
Awesome-Free-AI-Course
├── README.md
├── resources/                         # 📚 学习资料合集（外部链接汇总）
│   ├── deeplearning_material.md       #    AI / 深度学习资料总表
│   ├── writing_papers.md              #    论文写作资料
│   └── Awesome-Transformer-learning/  #    Transformer 专题资料（中 / 英）
└── notes/                             # 📝 学习笔记
    ├── LLM学习笔记/                     #    LLM 从零构建相关笔记
    │   ├── nanoGPT学习笔记/
    │   └── minimind学习笔记/
    └── Learning_Notes/                #    各类专题笔记
        ├── AutomataLearning/          #    混合自动机 / 混合系统辨识
        ├── GenAI/                     #    生成模型（VAE）
        └── Survival Analysis/         #    生存分析（Cox 模型）
```

## 🧭 内容导航

### 📚 学习资料合集 (`resources/`)

| 文件 | 说明 |
| --- | --- |
| [deeplearning_material.md](resources/deeplearning_material.md) | 所有学习资料的总表，包括 Python、深度学习、机器学习、强化学习、图神经网络、LLM、Diffusion / Flow 模型、数学基础、计算机基础、控制理论与深度学习等，不定期更新 |
| [Awesome-Transformer-learning](resources/Awesome-Transformer-learning) | Transformer 专题资料：结构讲解、子模块（位置编码、Softmax、缩放点积）、FlashAttention、Vision Transformer 等，提供 [中文版](resources/Awesome-Transformer-learning/Chinese_version.md) 和 [英文版](resources/Awesome-Transformer-learning/English_version.md) |
| [writing_papers.md](resources/writing_papers.md) | AI / ML 论文写作教程与建议 |

### 📝 学习笔记 (`notes/`)

#### LLM 学习笔记

| 内容 | 说明 |
| --- | --- |
| [nanoGPT 学习笔记](notes/LLM学习笔记/nanoGPT学习笔记) | Andrej Karpathy *Let's build GPT: from scratch* 视频的笔记，附 [带注释的 Colab 代码](notes/LLM学习笔记/nanoGPT学习笔记/gpt_dev_注释版.ipynb) 和模型结构图 |
| [minimind 学习笔记](notes/LLM学习笔记/minimind学习笔记) | 在 Mac (M1 Pro) 上跑通 [minimind](https://github.com/jingyaogong/minimind)（[笔记](notes/LLM学习笔记/minimind学习笔记/minimind_notes.md)）和多模态版 [minimind-v](https://github.com/jingyaogong/minimind-v)（[笔记](notes/LLM学习笔记/minimind学习笔记/minimind-v_notes.md)）的流程记录 |

#### 专题笔记

| 主题 | 内容 |
| --- | --- |
| Transformer 原理 | [缩放点积 (Scaled Dot-Product) 的数学推导](notes/Learning_Notes/scaling_dot_products.md) · [nn.Embedding 详解](notes/Learning_Notes/nn_Embedding.md) |
| 生成模型 | [VAE 教程 (Notebook)](notes/Learning_Notes/GenAI/VAE.ipynb) · [VQ-VAE 完整教程：从原理到实现](notes/Learning_Notes/vq_vae.md) |
| 概率与统计学习 | [高斯过程回归 (GPR) 教程](notes/Learning_Notes/gpr_tutorial.ipynb) · [SVD 笔记](notes/Learning_Notes/SVD.md) |
| 生存分析 | [基础概念](notes/Learning_Notes/Survival%20Analysis/survival_analysis.md) · [Cox 比例风险模型（中文）](notes/Learning_Notes/Survival%20Analysis/cox_cn.md) / [English](notes/Learning_Notes/Survival%20Analysis/cox.md) · [lifelines 实践 (Notebook)](notes/Learning_Notes/Survival%20Analysis/cox.ipynb) |
| 混合系统 / 自动机学习 | [混合自动机定义](notes/Learning_Notes/AutomataLearning/automata_defination/defination.md) · [基于夹角的聚类度量](notes/Learning_Notes/angle_clustering.md) · [混合系统辨识 Workshop (PDF)](notes/Learning_Notes/AutomataLearning/hybrid_systems_identification_workshop.pdf) · [弹跳球混合自动机动画 (HTML)](notes/Learning_Notes/AutomataLearning/bouncing-ball-animation.html) |
| 工程工具 | [使用 wandb 记录 MLP 训练的简单示例](notes/Learning_Notes/mlp_wandb_training.py) |

## 🤝 贡献指南

欢迎任何形式的贡献！

- **推荐资料**：发现了优质的免费 AI 学习资源，欢迎提交 [Issue](https://github.com/pengqianhan/Awesome-Free-AI-Course/issues) 或 Pull Request，补充到 [`resources/deeplearning_material.md`](resources/deeplearning_material.md) 中合适的分类下。
- **纠错**：发现链接失效、内容错误或笔记中的问题，欢迎直接指出或修正。
- **提交 PR 的步骤**：
  1. Fork 本仓库
  2. 新建分支：`git checkout -b add-some-resource`
  3. 提交修改：`git commit -m "docs: add xxx resource"`
  4. 推送分支并发起 Pull Request

推荐资料时请尽量注明：**资料名称 + 链接 + 一句话简介**（例如语言、难度、是否有中文字幕）。

## 🙏 致谢与声明

- 本仓库收集的课程、书籍、博客、代码等外部资料，版权归原作者所有，这里仅做整理和推荐。
- 部分笔记参考或改编自开源项目与教程（如 [nanoGPT](https://github.com/karpathy/nanoGPT)、[minimind](https://github.com/jingyaogong/minimind)、[GPR tutorial](https://github.com/jwangjie/Gaussian-Process-Regression-Tutorial) 等），出处已在对应笔记中注明。感谢这些作者的无私分享。

如果这个仓库对你有帮助，欢迎点一个 ⭐ Star 支持一下！
