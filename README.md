# Adaptive Zoning DES

## Project Overview 
Adaptive Zoning DES is a minimal, reproducible discrete-event simulator for studying **system-level bottlenecks** in large-scale humanoid robot fleets.
It models **Robot → Zone Controller → Central Supervisor** with queueing, network delay, and hotspot skew, focusing on **state dissemination latency** and overload behavior.

## Installation 
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if present
```

## Research Summary 
We study how **centralized aggregation** and **static zoning** create hotspots and overload under scale and skew.
We compare three schemes:
- **S0 Static Centralized** (zone is forward-only)
- **S1 Static Edge** (zone FIFO processing)
- **S2 Adaptive Hotspot-Aware Zoning** (dynamic reassignments)

Key metrics include **end-to-end latency**, **queue overload ratio**, and recovery behavior under hotspot injection.

## Usage 
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```

---

## Présentation du projet 
Adaptive Zoning DES est un simulateur à événements discrets minimal et reproductible pour étudier les **goulots d’étranglement système** dans des flottes de robots humanoïdes à grande échelle.
Il modélise **Robot → Contrôleur de zone → Superviseur central** avec files d’attente, latence réseau et concentration de charge, afin d’analyser la **latence de diffusion d’état** et la surcharge.

## Installation 
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # si présent
```

## Résumé scientifique 
Nous étudions comment **l’agrégation centralisée** et le **zoning statique** génèrent des points chauds et de la surcharge avec la montée en charge.
Nous comparons trois schémas :
- **S0 Centralisé statique** (zone relais)
- **S1 Edge statique** (traitement FIFO en zone)
- **S2 Zoning adaptatif sensible aux hotspots** (réaffectations dynamiques)

Les métriques principales sont la **latence de bout en bout**, la **surcharge de file d’attente** et la récupération après hotspot.

## Utilisation 
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```

---

## 项目简介 
Adaptive Zoning DES 是一个最小、可复现的离散事件仿真器，用于研究大规模人形机器人集群中的**系统级瓶颈**。
模型结构为 **机器人 → 区域控制器 → 中央监督器**，包含排队、网络延迟与热点倾斜，关注**状态传播延迟**与过载行为。

## 安装说明 
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # 若存在
```

## 研究摘要 
我们研究**集中式聚合**与**静态分区**在规模化和任务倾斜下引发热点与过载的问题。
对比三种方案：
- **S0 静态集中式**（Zone 仅转发）
- **S1 静态边缘式**（Zone FIFO 处理）
- **S2 热点自适应分区**（动态重分配）

关键指标包括**端到端延迟**、**队列过载比例**以及热点恢复行为。

## 使用指南 
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```
