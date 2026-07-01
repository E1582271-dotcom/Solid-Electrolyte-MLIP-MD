# 项目二（★旗舰，最高信号）：硫化物 MLIP + MD 离子电导率 pipeline

**定位**：深度旗舰，直接命中国内硫化物固态电池计算组的核心方法栈。计划周 W5–W9，里程碑②。
`notebooks/00_motivation_run.ipynb` = 现成的 MACE-MP+ASE+Langevin MD hello-world（从旧工作区抢救，W5 起点）。

## 目标体系
**Li6PS5Cl（argyrodite 硫银锗矿）为主**，Li10GeP2S12(LGPS) 或 β-Li3PS4 为辅。

## MVP
下载 MACE-MP-0（或 MatterSim）→ 用少量 DFT 数据对硫化物微调 → ASE/LAMMPS 跑多温（600/800/1000K）NVT MD → kinisi / pymatgen `DiffusionAnalyzer` 算 MSD、D、Arrhenius 外推 300K σ → 对比实验。

**对标实验靶值**：
- Li6PS5Cl 室温 σ ≈ 3.15×10⁻³ S/cm（优化烧结，ACS AMI acsami.8b15121）；机械化学 ~1.33 mS/cm；Li6PS5X(X=Cl,Br,I) 族室温普遍 >1 mS/cm。

## 进阶
复现 Cl 过量 / Al 掺杂提升趋势，对标 DPA-SSE。靶值：
- Li5.4Al0.1PS4.7Cl1.3 室温 7.29×10⁻³ S/cm（Li6PS5Cl 的 ~4.7×，Eₐ~0.09 eV）
- Li5.5PS4.5Cl1.5 ~9.4 mS/cm；Li5.7PS4.7Cl1.3 ~6.4 mS/cm

## 工具栈
MACE / MatterSim / CHGNet + ASE + (LAMMPS) + pymatgen-analysis-diffusion + kinisi + DeePMD-kit（可选，DPA 对比）。

## 算力
微调几小时（Colab/AutoDL）；多温生产 MD 各几 ns，**W8 按需租 AutoDL RTX 5090**（见 `../compute/COMPUTE_NOTES.md`）。

## 常见陷阱（写进技术报告）
1. 不微调直接用 universal MLIP → 系统性高估/低估 σ 和体积（实证偏差 2–40%）。**故设计「先 baseline 后微调」。**
2. MD 时间太短 → MSD 统计不收敛、Arrhenius 拟合噪声大。
3. 高温外推室温的**非 Arrhenius 行为**（argyrodite 有此现象）。
4. 无序结构（Cl/S 在 4c/4a 位混排）初始构型枚举不当。
5. Nernst-Einstein 忽略离子关联；PES 软化致势垒偏低；晶界使实测 σ 下降。

## 可复用资产
- Li2CO3_ML 项目的 EI/主动学习脚本 → 微调训练集 QBC 主动采样（留给 W7）。
- 离子膜传输直觉（MSD/扩散/迁移数/Arrhenius/Eₐ）→ 输运链路直接复用。

## checkpoint（里程碑②）
(a) 讲清「我的 D 为何有误差棒、误差从哪来、凭什么信/不信这个 σ、MLIP 大致在学什么」；
(b) 多温 Arrhenius 图（带误差棒）+ σ(300K) 与实验同数量级 + 对标讨论 + 可复跑 repo。

---

## 已实现：W5–W6 baseline pipeline（双势对比，薄管道）

线性编号脚本 + 可复用 `src/` 模块，沿用项目一约定。**两个未微调通用势（MACE-MP-0 与 MatterSim）跑在同一 Li₆PS₅Cl 胞上对照**——这是"先 baseline 后微调"的诚实第一步。

| 脚本 | 作用 | 跑在哪 |
|---|---|---|
| `src/structure.py` + `01_build_structure.py` | 拉真实 **MP mp-985592**（缓存 CIF）→ 按 P–S 键长识别自由阴离子 → S/Cl 半占据 → **Ewald 能量枚举**有序近似 → `data/config*.cif` + 概览图 | 本机 CPU |
| `src/md.py` + `02_baseline_md.py` | calculator-agnostic NVT Langevin MD（`--mlip mace\|mattersim\|both`），多温、带 checkpoint → `data/traj/<mlip>_<T>K.traj` | Colab T4 |
| `src/transport.py` + `03_analyze_transport.py` | **pymatgen DiffusionAnalyzer**（主干，D+σ）+ **kinisi**（误差棒）→ Nernst-Einstein σ(T) → Arrhenius → 对标实验 → `metrics.json` + 图 | CPU/Colab |
| `notebooks/01_baseline_md.ipynb` | Colab 编排：装重包 + 依次跑 01→02→03 | Colab T4 |

```bash
# 本机（结构构建，纯 pymatgen；用绝对路径调 venv 避免 3.14 的 sys.prefix 警告）
~/Code/AI4SSB/.venv/bin/python 01_build_structure.py
# Colab（MD + 分析，重包在 GPU 上）—— 见 notebooks/01_baseline_md.ipynb
python 02_baseline_md.py --temps 600,800,1000 --steps 50000   # 薄：先 30000、2 温点验证
python 03_analyze_transport.py
```
本机环境：`~/Code/AI4SSB/.venv`（**Python 3.14.6**，uv 建）。重 MLIP 包（torch/mace/mattersim）不在本机，MD 在 Colab 跑。

**结构来源**：MP mp-985592（Li₆PS₅Cl, F-43m, 有序 DFT 近似），常规立方胞 52 原子（24 Li），a=10.28 Å。S/Cl 自由阴离子无序从有序胞反推 + Ewald 枚举（config0=基态排列、config1=位点无序变体），**零硬编码坐标**。

## Honest limitations（baseline，写进技术报告）
- **未微调**：通用 MLIP 直接用 → 体积/D/σ 预期 2–40% 系统偏差（*故意先看*，微调=W7）。
- **薄管道**：MD 短、**MSD 统计未收敛** → σ 仅数量级指示，非定量。
- **MP/PBE 晶格** a=10.28 Å 比实验 ~9.86 Å 大 ~4.3% → 体积偏差传到 D、σ。
- **Nernst-Einstein** 忽略离子关联（未做 Haven 比修正）。
- **单晶上限**：无晶界，真实多晶 σ 应更低。
- **非 Arrhenius**：argyrodite 有此现象，线性外推 300K 引入误差。
- **无序采样**：仅几个代表构型，非完整 S/Cl 系综。

## 结果（W6 基线 + W8 生产 + W7 微调，来自 `data/metrics{,_long,_prod,_ft}.json`）
对标实验：Li₆PS₅Cl 室温 ~3.15 mS/cm（烧结）/ 1.33 mS/cm（机械化学）。D@1000K 为 kinisi 估计（带误差棒）。

| MLIP（胞·时长） | D@1000K (cm²/s) | σ(300K) 外推 (mS/cm) | Eₐ (eV) | 与实验 3.15 倍率 | Arrhenius R² | 备注 |
|---|---|---|---|---|---|---|
| MACE-MP-0 (52-atom, 50 ps) | 5.0×10⁻⁵ | 29.6 | 0.198 | 9.4× | 0.81 | 未微调基线；低温 MSD 未收敛 |
| MACE-MP-0 (52-atom, 150 ps) | 4.6×10⁻⁵ | 12.9 | 0.232 | 4.1× | 0.999 | 未微调；已收敛，Arrhenius 成直线 |
| MatterSim (52-atom, 50 ps) | 4.3×10⁻⁵ | 16.0 | 0.211 | 5.1× | 0.85 | 未微调；薄数据 |
| **MatterSim (52-atom, 150 ps)** | 3.8×10⁻⁵ | **2.06** | 0.286 | **0.65×** | 0.983 | 未微调；已收敛；**所有跑法里最接近实验** |
| **MACE-MP-0 (416-atom, 200 ps)** | 5.5×10⁻⁵ | **5.57** | 0.265 | **1.8×** | 0.999 | **W8 生产；基座未微调；2×2×2 超胞** |
| **MACE-微调 (416-atom, 200 ps)** | 3.0×10⁻⁵ | **0.497** | 0.331 | **0.16×** | 0.998 | **W7 微调(SWA)；27 个 QE-Γ 力标注；验证力 RMSE 234→62 meV/Å** |

**读出**：
- **未微调基线（52-atom）**：两个通用势 50 ps 时都系统性过预测单晶 σ（无晶界上界），MatterSim 比 MACE 轻一档；延长到 150 ps 两者都收敛下移——MACE 29.6→12.9（9.4×→4.1×）、R² 0.81→0.999，**MatterSim 16.0→2.06（5.1×→0.65×）、R² 0.85→0.983，收敛后反而是所有跑法里最接近实验的**——证实短跑的过预测/低温摆动是 MSD 采样噪声而非真·非阿伦尼乌斯。
- **W8 生产（416-atom·200 ps·基座）**：更大胞 + 更长时间把 σ 收到 **5.57 mS/cm（1.8× 实验）**、R²=0.999、能量漂移 0.007 meV/atom/ps——纯 MLIP-MD 已达文献级一致。这是里程碑②核心交付。
- **W7 微调（27 个 QE-Γ DFT 力标注 → SWA 微调，验证力 RMSE 234→62 meV/Å）**：势变硬（Eₐ 0.265→0.331 eV），σ 降到 **0.497 mS/cm（0.16×，欠预测 ~6×）**。基座**过预测**、微调**欠预测**恰把实验夹在中间（几何均值 ~1.7 mS/cm）；偏硬方向大概率源于 **Γ-only k 采样 + GBRV 赝势 + 27 构型小数据集**——诚实写进技术报告，作为"标注质量 → 势 → σ"敏感性的实证，也是下一步（更密 k 点 / 更多含迁移态构型）的动机。
