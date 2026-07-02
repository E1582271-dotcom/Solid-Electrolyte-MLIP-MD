# 项目二技术报告：Li₆PS₅Cl 的 MLIP-MD 离子电导率

**里程碑 ②（W5–W9）· 硫化物固态电解质 · 通用机器学习势 + 分子动力学 + 微调**

---

## 1. 目标与体系

用机器学习原子间势（MLIP）驱动分子动力学（MD），计算硫银锗矿型固态电解质 **Li₆PS₅Cl** 的 Li⁺ 离子电导率 σ(300 K)，并与实验（烧结 ~3.15 mS/cm；机械化学 ~1.33 mS/cm）对标。

完整链路：
> 结构建模 → 多温 NVT-MD → 均方位移 MSD/扩散系数 D → Nernst-Einstein σ(T) → Arrhenius 外推 σ(300 K) → 对标实验 → **DFT 标注 + 微调**再评估。

设计原则：**「先 baseline 后微调」**——先用未微调的通用势看系统偏差，再用少量 DFT 数据微调，量化"标注质量 → 势 → σ"的敏感性。

---

## 2. 方法

### 2.1 结构建模（`01_build_structure.py`, `src/structure.py`）
- 拉取 Materials Project **mp-985592**（缓存 CIF）。Li₆PS₅Cl 硫银锗矿：PS₄³⁻ 四面体骨架 + 游离 S²⁻/Cl⁻ 在 4a/4d 位半占据。
- 按 P–S 键长识别游离阴离子，对 S/Cl 半占据做 **Ewald 能量枚举**取有序近似 → `data/config0.cif`（52 原子，晶胞 a≈10.28 Å）。
- 生产用 **2×2×2 超胞** `data/config0_sc222.cif`（416 原子，密度不变 1.641 g/cm³，192 个 Li）——更大盒子压低 MSD 统计噪声与有限尺寸效应。

### 2.2 分子动力学（`02_baseline_md.py`, `src/md.py`）
- ASE **NVT Langevin**，摩擦系数 0.01 fs⁻¹，时间步 1 fs，温度 600/800/1000 K（高温加速扩散以在可负担时长内取样，再 Arrhenius 外推到 300 K）。
- 势：**MACE-MP-0 (small, L0)** 与 **MatterSim (v1.0.0-1M)** 两个通用势对照；float32，NVIDIA A40 GPU。
- 平衡后记录，每 50 步存一帧。时长分档：50 ps（薄）/ 150 ps（收敛检验）/ **200 ps（生产，416 原子）**。

### 2.3 输运分析（`03_analyze_transport.py`, `src/transport.py`）
- **pymatgen `DiffusionAnalyzer`** 作主干算 D（从 MSD 线性段）。
- **kinisi** 作误差棒：用广义最小二乘 + bootstrap 处理 MSD 时间点间的强相关，给出 D 的统计不确定度（见 §4）。
- **Nernst-Einstein**：σ = n q² D / (k_B T)，n 为 Li⁺ 数密度（从结构与体积算）。
- **Arrhenius**：拟合 ln(σT) = −E_a/(k_B T) + c（教科书直线形式），外推 σ(300 K)、给 E_a 与 R²。

### 2.4 DFT 标注（`finetune/11_label_qe.py`, `label_qe.pbs`, `submit_shards.sh`）
- 从 MD 轨迹采样 **27 个快照**（52 原子），用 **Quantum ESPRESSO pw.x** 做 SCF 单点，产出能量+力（+应力）。
- 赝势 **GBRV all-PBE 超软**（Li/P/S/Cl），ecutwfc 60 / ecutrho 480 Ry，gaussian smearing degauss 0.01，conv_thr 1e-7，`tprnfor+tstress`。
- **Γ 点实波函数**采样（`--kpts gamma`）：相比 2×2×2 便宜 ~16×，对 delta 微调标注足够；单点 SCF ~26 min（16 核，Intel-MPI，`npool=1`）。
- 27 个快照拆成 **27 个单配置 CPU 作业并发**标注（`--start` 偏移分片），~1.8 h 全部完成，0 失败。

### 2.5 微调（`finetune/20_finetune_mace.sh`, `finetune_mace.pbs`）
- `mace_run_train`，`--foundation_model` 指向本地缓存的 MACE-MP-0 small（离线安全）。
- **`--E0s=average`（关键）**：QE 绝对能量 ~−269 eV/atom 与 MACE-MP 参考尺度不同，必须从本数据重算每元素参考能，否则能量损失被巨大常数偏移主导；力与 E0 无关故不受影响，且力权重 10 ≫ 能量权重 1。
- lr 1e-4，batch 4，120 epoch + **SWA**，float32，A40。27 构型 → 22 训练 / 5 验证。

---

## 3. 结果

对标实验：Li₆PS₅Cl 室温 ~3.15 mS/cm（烧结）。图见 `figures/03_arrhenius{,_long,_prod,_ft}.*`（双栏 Nature 规格，MD 点带 kinisi 误差棒）、`03_sigma300_vs_expt*.*`。

| 模型（胞·时长） | D@1000K (cm²/s) | σ(300K) (mS/cm) | E_a (eV) | 与实验倍率 | R² |
|---|---|---|---|---|---|
| MACE-MP-0 (52-atom, 50 ps) | 5.0×10⁻⁵ | 29.6 | 0.198 | 9.4× | 0.81 |
| MACE-MP-0 (52-atom, 150 ps) | 4.6×10⁻⁵ | 12.9 | 0.232 | 4.1× | 0.999 |
| MatterSim (52-atom, 50 ps) | 4.3×10⁻⁵ | 16.0 | 0.211 | 5.1× | 0.85 |
| **MatterSim (52-atom, 150 ps)** | 3.8×10⁻⁵ | **2.06** | 0.286 | **0.65×** | 0.983 |
| **MACE-MP-0 (416-atom, 200 ps)** 生产 | 5.5×10⁻⁵ | **5.57** | 0.265 | **1.8×** | 0.999 |
| **MACE-微调 (416-atom, 200 ps)** | 3.0×10⁻⁵ | **0.497** | 0.331 | **0.16×** | 0.998 |

微调势验证力 RMSE：基座 234 → 微调 **62.4 meV/Å**（3.75× 改善），验证能 RMSE 3.3 meV/atom。收敛过程见 `figures/04_finetune_convergence.*`（验证 RMSE 力/能 vs epoch,对比基座基线,~epoch 90 的尖峰为 SWA 第二阶段重启）。

MD 稳定性（守恒性检查,温度与能量随时间平稳、无爆炸/漂移）见 `figures/02_md_stability_mace{,_prod,_ft}.*`（基线 50ps / 生产 200ps·416 原子 / 微调 200ps）。

### 3.1 文献对标（结果评估）

| 来源 | 类型 | σ(300 K) mS/cm | Eₐ (eV) | 方法 / 条件 |
|---|---|---|---|---|
| **本工作（生产）** | MLIP-MD | **5.57** | **0.265** | MACE-MP-0，416 原子，200 ps，600–1000 K 外推 |
| 本工作（MatterSim 150 ps） | MLIP-MD | 2.06 | 0.286 | MatterSim，52 原子，150 ps |
| 本工作（微调） | MLIP-MD | 0.497 | 0.331 | Γ-DFT 微调 MACE |
| 文献（计算） | MLIP-MD | 2.2（800–1200 K 外推）/ 0.22（500–700 K 外推） | — | 400–1200 K；作者报告 300 K/100 ns **无跳跃**须外推，且外推值随拟合温区强烈变化 [1] |
| 实验（优化烧结） | EIS | 3.15 | — | 550 °C 烧结 [2] |
| 实验（液相合成） | EIS | >2 | — | [3] |
| 实验（湿磨+退火） | EIS | 1.0–1.9 | — | [4] |
| 实验（机械化学） | EIS | 1.33 | — | 球磨 |
| 实验（汇总范围） | EIS | ~1–3.2 | **0.22–0.38** | 多工艺；溶剂处理低至 ~0.20–0.25，部分研究 0.35–0.38 [5] |

**读出**：
- **σ(300 K)**：生产值 5.57 = 优化烧结实验（3.15）的 **1.8×**，与其他 MLIP-MD 高温外推（2.2）同量级；全部在"实验值上方约 2 倍"内——对纯未微调势属很好的一致。
- **Eₐ**：生产 0.265 eV、微调 0.331 eV **均落在实验 0.22–0.38 eV 区间**，温度依赖抓对。
- **外推敏感性有文献印证**：[1] 明确报告 300 K 下 100 ns 无跳跃、必须从高温外推，且外推值随拟合温区从 0.22 变到 2.2 mS/cm——与本工作"长程外推有额外不确定性、故 50→150 ps 收敛很关键"一致。
- **单晶上界**：实验多为多晶含晶界，模拟为单晶,故"略高于典型实验"符合预期；微调欠预测（0.16×）为离群点，归因见 §5。

参考：[1] arXiv:2403.14116 · [2] ACS Appl. Mater. Interfaces 2018, `10.1021/acsami.8b15121` · [3] Chem. Mater. 2022, `10.1021/acs.chemmater.2c03818` · [4] Front. Chem. 2021, `10.3389/fchem.2021.778057` · [5] ACS Appl. Mater. Interfaces, `10.1021/acsami.8b07476`。

---

## 4. 误差分析（「我的 D 为何有误差棒、误差从哪来」）

**统计误差（图中误差棒）**：MSD(t) 各时间点由同一条轨迹的滑窗平均得到，**彼此强相关**——直接对 MSD 做普通最小二乘会低估 D 的方差。kinisi 用**广义最小二乘 + bootstrap** 显式建模这一相关，给出 D 的后验分布，其标准差即误差棒。150/200 ps 收敛轨迹误差棒小（统计足），50 ps 低温点误差棒大且 R² 差（MSD 未进入线性扩散段，采样不足）——这正是把 50→150 ps 后 600 K 点归位、R² 0.81→0.999 的原因：**低温摆动是采样噪声，不是真·非阿伦尼乌斯**。

**系统误差（不进误差棒，但决定可信度）**：
1. **晶格偏大**：MP/PBE 晶格 a≈10.28 Å 比实验 ~9.86 Å 大 ~4.3%，格子松 → 扩散偏快 → σ 偏高。
2. **单晶上限**：模拟无晶界，真实多晶 σ 更低——MD 给的是**上界**。
3. **Nernst-Einstein 近似**：忽略离子间关联（Haven 比 H_R≠1），系统性影响 σ 绝对值。
4. **势的 PES 软化 / 硬化**：通用势对硫化物势垒可偏低（σ 偏高）；微调若过硬则势垒偏高（σ 偏低）——见 §5。
5. **有限尺寸 / 时长**：52 原子已用 416 原子超胞与 200 ps 缓解。

**凭什么信/不信**：可信的是**数量级与趋势**（Arrhenius 直线 R²>0.99、E_a 落在文献 0.2–0.35 eV、生产值 1.8× 实验）；不宜过读的是**绝对值到 2 位有效数字**（受上述系统误差与 NE 近似影响）。

---

## 5. 讨论（「MLIP 大致在学什么」+ 对标）

- **未微调基座过预测，但随采样收敛下移**：两个通用势 50 ps 时都系统性高估单晶 σ（MatterSim 比 MACE 轻一档）。它们在 MP 的 PBE 平衡态数据上训练，学到的是**近平衡的能量-力面**；对 Li⁺ 迁移过渡态与硫化物软骨架的细节外推不足，倾向低估势垒 → 高估 σ。延长到 150 ps 后两者都明显收敛：MACE 4.1×、**MatterSim 降到 0.65×（σ 2.06 mS/cm，所有跑法里最接近实验）**——说明短跑的高估很大部分是 MSD 未收敛的统计伪影。**W8 生产**（416 原子 + 200 ps）把 MACE 收到 **1.8× 实验、R²=0.999**，是纯 MLIP-MD 能达到的文献级一致。
- **微调把势变硬**：用 27 个 QE-Γ DFT 力微调后，验证力 RMSE 从 234 降到 62 meV/Å（模型确实更贴我们的 DFT 力），但 **E_a 0.265→0.331 eV、σ 降到 0.16×**。即：**基座过预测、微调欠预测,恰把实验夹在中间**（几何均值 ~1.7 mS/cm ≈ 0.5× 实验）。
- **偏硬方向的来源**（诚实归因）：(a) **Γ-only k 采样** 对 52 原子胞欠采布里渊区，可系统性偏置力/能；(b) **GBRV/PBE** 与 MACE-MP 的 VASP/PBE-PAW 参考不同，微调把力"重定标"到我们的 DFT 风味；(c) **27 构型小数据集**、且多为近平衡快照，缺少迁移过渡态构型 → 过拟合到偏硬的局域势面。这说明 σ 对**标注质量高度敏感**——是本项目最有价值的实证结论之一。

---

## 6. 诚实局限（写进报告）
- 通用势未微调直接用 → 系统偏差（本项目实测过预测 1.8–9.4×）。
- 薄管道时 MSD 未收敛 → σ 仅数量级指示（已用 150/200 ps 缓解）。
- MP/PBE 晶格偏大 4.3%；单晶无晶界（上界）；NE 忽略关联。
- 微调数据小（27）、Γ-only、GBRV——σ 偏硬，**非"微调必然更准"**，而是"标注质量决定一切"的反例。

---

## 7. 可复现

```bash
# 环境：MACE/MatterSim + ASE + pymatgen-analysis-diffusion + kinisi（见 finetune/README.md）
python 01_build_structure.py                       # 结构 + 超胞
python 02_baseline_md.py --temps 600,800,1000 --steps 200000 --supercell-tag _sc222 --traj-tag _prod
python 03_analyze_transport.py --traj-tag _prod    # D/σ/Arrhenius + 图 + metrics_prod.json
# 微调（Vanda A40/CPU，见 finetune/）：
bash finetune/submit_shards.sh                     # 27 个 QE-Γ DFT 标注
python finetune/12_split_train_valid.py            # 22 训练 / 5 验证
qsub  finetune/finetune_mace.pbs                   # A40 微调 → li6ps5cl_ft_stagetwo.model
python 02_baseline_md.py --mace-model finetune/models/li6ps5cl_ft_stagetwo.model --traj-tag _ft ...
python 03_analyze_transport.py --traj-tag _ft
```

- 关键数据：`data/metrics{,_long,_prod,_ft}.json`、`finetune/data/labelled.xyz`（27 个 DFT 标注）、`finetune/results/`（训练曲线）。
- 模型权重按 `.gitignore` 不入库，可从 `labelled.xyz` + 微调脚本复现。
- 所有计算跑在 **NUS Vanda**（个人免费额度，A40 MD / CPU DFT），无付费租用。

---

*生成：2026-07-01。数据溯源见各 `metrics_*.json` 与 `finetune/results/`。*
