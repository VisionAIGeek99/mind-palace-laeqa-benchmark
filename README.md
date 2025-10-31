# Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering

[[paper](https://www.arxiv.org/pdf/2507.12846)]
[[project](https://mind-palace-laeqa.github.io/)]


## Abstract
As robots become increasingly capable of operating over extended periods—spanning days, weeks, and even months—they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration.

To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determine when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency.

## Long-term Active EQA Benchmark
We introduce the **Long-term Active EQA (LA-EQA) Benchmark** to evaluate an agent’s ability to understand an environment—and track how it changes—over **days to months**.

We release the LA-EQA question set and the scene files:

1. **List of questions:** [`la_eqa_benchmark/eqa_questions.json`](la_eqa_benchmark/eqa_questions.json)  
2. **Scene files (≈6 GB):** [Google Drive download](https://drive.google.com/file/d/1lNcNgpzr9SnJR1ye7vBXpWeAO-YwD14B/view?usp=sharing)


## Mind Palace Exploration for Long-term Active EQA

This repository provides the reference implementation of **Mind Palace Exploration**, our algorithm for **long-term exploration and memory-based reasoning** in LA-EQA.  


### Installation

Requires **Python ≥ 3.9**.

```bash
conda env create -f environment.yaml
conda activate laeqa
pip install -e .
```

Installing Recognize Anything Model (RAM) - Optional
Please follow the installation instruction in https://github.com/xinyu1205/recognize-anything

### Running One Example

To run a single demonstration of **Mind Palace Exploration**, open the following notebook:

[`run_la_eqa_mind_palace_exploration.ipynb`](scripts/run_la_eqa_mind_palace_exploration.ipynb) 

The implementation uses the **OpenAI API** for LLM and VLM.  
Before running the notebook, please set your OPENAI_API_KEY key as an environment variable:


### Running the whole evaluation
To evaluate the entire LA-EQA question set, open the same notebook and modify the following flag:

```python
test_one_question = False
```

## Citing Mind Palace Exploration and LA-EQA Benchmark

```tex
@inproceedings{ginting2025laeqa,
  author={ Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna,
Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim,
Mykel J. Kochenderfer, Ali-akbar Agha-Mohammadi, Shayegan Omidshafiei},
  title={{Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering}},
  booktitle={{Conference on Robot Learning (CoRL)}},
  year={2025},
}
```
