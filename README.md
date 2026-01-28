# Adaptive Zoning DES

## Project Overview (EN)
Adaptive Zoning DES is a minimal, reproducible discrete-event simulator for studying **system-level bottlenecks** in large-scale humanoid robot fleets.
It models **Robot ? Zone Controller ? Central Supervisor** with queueing, network delay, and hotspot skew, focusing on **state dissemination latency** and overload behavior.

## Installation (EN)
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if present
```

## Research Summary (EN)
We study how **centralized aggregation** and **static zoning** create hotspots and overload under scale and skew.
We compare three schemes:
- **S0 Static Centralized** (zone is forward-only)
- **S1 Static Edge** (zone FIFO processing)
- **S2 Adaptive Hotspot-Aware Zoning** (dynamic reassignments)

Key metrics include **end-to-end latency**, **queue overload ratio**, and recovery behavior under hotspot injection.

## Usage (EN)
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```

---

## Pr?sentation du projet (FR)
Adaptive Zoning DES est un simulateur ? ?v?nements discrets minimal et reproductible pour ?tudier les **goulots d??tranglement syst?me** dans des flottes de robots humano?des ? grande ?chelle.
Il mod?lise **Robot ? Contr?leur de zone ? Superviseur central** avec files d?attente, latence r?seau et concentration de charge, afin d?analyser la **latence de diffusion d??tat** et la surcharge.

## Installation (FR)
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # si pr?sent
```

## R?sum? scientifique (FR)
Nous ?tudions comment **l?agr?gation centralis?e** et le **zoning statique** g?n?rent des points chauds et de la surcharge avec la mont?e en charge.
Nous comparons trois sch?mas :
- **S0 Centralis? statique** (zone relais)
- **S1 Edge statique** (traitement FIFO en zone)
- **S2 Zoning adaptatif sensible aux hotspots** (r?affectations dynamiques)

Les m?triques principales sont la **latence de bout en bout**, la **surcharge de file d?attente** et la r?cup?ration apr?s hotspot.

## Utilisation (FR)
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```

---

## ???? (??)
Adaptive Zoning DES ??????????????????????????????????**?????**?
????? **??? ? ????? ? ?????**??????????????????**??????**??????

## ???? (??)
```bash
git clone git@github.com:yezhan14promax/adaptive-zoning-des.git
cd adaptive-zoning-des
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # ???
```

## ???? (??)
????**?????**?**????**?????????????????????
???????
- **S0 ?????**?Zone ????
- **S1 ?????**?Zone FIFO ???
- **S2 ???????**???????

??????**?????**?**??????**?????????

## ???? (??)
```bash
python -m arm_sim.experiments.demo_s0
python -m arm_sim.experiments.demo_s1
python -m arm_sim.experiments.run_hotspot --mode main
python -m arm_sim.plots.make_paper_figures
```
