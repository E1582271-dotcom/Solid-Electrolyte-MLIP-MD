"""生成合并版 00_motivation_run.ipynb：装包 → LiCl 单点能 → 600K MD → 弛豫 → 真实 Li6PS5Cl 单点能。
用 json 写,保证 UTF-8 + 合法 JSON,避免复制粘贴乱码。一个文件上传 Colab 全部运行即可。"""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# P0 动机 run — 你的第一个 ML 势工作流(一个文件全流程)

一次跑完四件事,建立"它真的能跑"的成就感 + 摸到后面算电导率要用的全部基本功:
1. **单点能**:用预训练通用势 MACE-MP 给晶体算能量
2. **分子动力学(MD)**:600K Langevin 恒温,看温度被稳住、能量不发散
3. **结构弛豫**:把晶体揉皱,看 MLIP 沿受力把它推回低能量构型
4. **真实体系**:从 Materials Project 拉真实 Li₆PS₅Cl,算单点能 —— 旗舰项目二的入口

**用法**:Colab → `代码执行程序` → `更改运行时类型` → **T4 GPU** → 上传本文件 → `代码执行程序` → **全部运行**。
**前置**:左侧 🔑 Secrets 里建好 `MP_API_KEY`(值=你的 MP key)并对本 notebook 开启「笔记本访问」。
**血泪经验**:① 运行时必须切 GPU(cell 1 要 `CUDA: True`);② hello-world 用 float32,别用 float64(T4 双精度只有 1/32 算力)。""")

code("""# 1) 装包 + 确认 GPU(约 1-2 分钟)
!pip install -q mace-torch mp-api ase pymatgen
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print('CUDA:', torch.cuda.is_available(), '| GPU:', gpu, '| device:', device)
# 若是 False/CPU:更改运行时类型 → T4 GPU → 全部运行""")

code("""# 2) 建小晶体 + 挂上 MACE-MP 通用势,算单点能
from ase.build import bulk
from mace.calculators import mace_mp

calc = mace_mp(model='small', dispersion=False, default_dtype='float32', device=device)  # 这个势对象后面复用
atoms = bulk('LiCl', 'rocksalt', a=5.14) * (3, 3, 3)   # 3x3x3 超胞 = 216 原子
atoms.calc = calc
print(len(atoms), '个原子')
print('势能 =', round(atoms.get_potential_energy(), 3), 'eV')   # MACE-MP 一行算出能量""")

md("""## (2) 分子动力学:600K 跑 2 ps,看恒温器把温度稳住

- 一开始温度会**腰斩**(能量均分:动能往势能分一半);Langevin 恒温器再慢慢把它焐回 600K。
- 温度大幅振荡是**真实物理**(有限尺寸热涨落 σ_T/T = √(2/3N)),不是噪声 —— 体系越大抖得越小。""")

code("""# 3) 600K NVT(Langevin)MD,2000 步 x 1 fs = 2 ps,每 200 步打印进度
import time
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(atoms, temperature_K=600)
dyn = Langevin(atoms, timestep=1.0*units.fs, temperature_K=600, friction=0.01)

steps, E, T = [], [], []
t0 = time.time()
def log():
    s = dyn.get_number_of_steps()
    steps.append(s); E.append(atoms.get_potential_energy()/len(atoms)); T.append(atoms.get_temperature())
    if s % 200 == 0:
        print(f"  step {s:4d}/2000  T={T[-1]:6.1f}K  E/atom={E[-1]:.4f}eV  ({time.time()-t0:.0f}s)")
dyn.attach(log, interval=10)
dyn.run(2000)
print('完成,用时', round(time.time()-t0,1), '秒')""")

code("""# 4) 出图:温度应在 600K 附近大幅振荡、每原子能量应稳定不发散
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(steps, T); ax[0].axhline(600, ls='--', c='r'); ax[0].set(xlabel='step', ylabel='T (K)', title='Temperature')
ax[1].plot(steps, E); ax[1].set(xlabel='step', ylabel='E/atom (eV)', title='Potential energy')
plt.tight_layout(); plt.show()""")

md("""## (3) 结构弛豫:揉皱 → 让 MLIP 沿受力推回去

弛豫 = 沿着力(force = -∇E)把原子挪到能量极小点。MD 之外的另一半基本功。""")

code("""# 5) 弛豫:故意把晶体揉皱,再让 MLIP 推回低能量构型
from ase.optimize import BFGS

relax_atoms = bulk('LiCl', 'rocksalt', a=5.14) * (2, 2, 2)
relax_atoms.calc = calc                        # 复用同一个 MACE-MP 势
E0 = relax_atoms.get_potential_energy()
relax_atoms.rattle(stdev=0.15, seed=0)         # 每个原子加 ~0.15 Å 随机位移
E_rattled = relax_atoms.get_potential_energy()
BFGS(relax_atoms, logfile=None).run(fmax=0.02, steps=100)   # 弛豫到最大受力 < 0.02 eV/Å
E_relaxed = relax_atoms.get_potential_energy()

print(f'理想晶体 E0        = {E0:.3f} eV')
print(f'揉皱后   E_rattled = {E_rattled:.3f} eV   (升高 {E_rattled - E0:+.3f})')
print(f'弛豫后   E_relaxed = {E_relaxed:.3f} eV   (回落到 E0 附近 = MLIP 把结构推回去了)')
print(f'弛豫收敛 fmax = {abs(relax_atoms.get_forces()).max():.4f} eV/Å')""")

md("""## (4) 真实体系:从 Materials Project 拉 Li₆PS₅Cl,算单点能

正式接上旗舰项目二。Li₆PS₅Cl 是无序结构(Cl/S 在 4a/4c 位无序),今天先用最稳的**有序近似**;W5 再正式处理无序。""")

code("""# 6) 从 Materials Project 拉真实 Li6PS5Cl 结构
import os
from mp_api.client import MPRester

try:
    from google.colab import userdata
    MP_API_KEY = userdata.get('MP_API_KEY')          # 从 Colab Secret 读,不硬编码
except Exception:
    MP_API_KEY = os.environ.get('MP_API_KEY')
assert MP_API_KEY, "没读到 MP_API_KEY:检查左侧 Secrets 里是否建了 MP_API_KEY 并对本 notebook 开启访问"

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        formula="Li6PS5Cl",
        fields=["material_id", "formula_pretty", "energy_above_hull", "symmetry", "structure"])

print(f"找到 {len(docs)} 个 Li6PS5Cl 条目")
ordered = [d for d in docs if d.energy_above_hull is not None and d.structure.is_ordered]
print(f"其中有序(可直接跑)的有 {len(ordered)} 个:")
for d in sorted(ordered, key=lambda x: x.energy_above_hull):
    sg = d.symmetry.symbol if d.symmetry else '?'
    print(f"  {d.material_id:12s}  E_hull={d.energy_above_hull:.3f} eV/atom  {sg}  {len(d.structure)} 原子")

if ordered:
    best = min(ordered, key=lambda x: x.energy_above_hull)
    struct = best.structure
    print()
    print(f"→ 选中最稳定的有序近似: {best.material_id}  ({len(struct)} 原子/原胞)")
else:
    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
    valid = [d for d in docs if d.energy_above_hull is not None]
    best = min(valid, key=lambda x: x.energy_above_hull)
    print()
    print(f"全是无序条目,对最稳的 {best.material_id} 做有序化...")
    struct = OrderDisorderedStructureTransformation().apply_transformation(best.structure)
    print(f"→ 有序化完成: {len(struct)} 原子/原胞")""")

code("""# 7) 给真实 Li6PS5Cl 算 MACE-MP 单点能(和上面 LiCl 同一套机器,只是体系换成旗舰目标)
real = struct.to_ase_atoms()
real.calc = calc                               # 复用同一个 MACE-MP 势
e = real.get_potential_energy()
print("化学式 =", real.get_chemical_formula(), "| 原子数 =", len(real))
print(f"MACE-MP 单点能 = {e:.3f} eV  ({e/len(real):.4f} eV/atom)")
print(f"最大受力 fmax  = {abs(real.get_forces()).max():.3f} eV/Å  (未弛豫,不为 0 正常)")""")

md("""## 全跑通了 = 环境就绪 + 四项基本功到手

你已经能:**算单点能 / 跑 MD / 做弛豫 / 从数据库拉真实结构喂给 ML 势**。这就是 capstone 算 Li⁺ 电导率的同一套机器。

**下一步(W5)**:正式处理 Li₆PS₅Cl 的无序 —— 枚举构型、多温 MD、从轨迹算 Li⁺ 扩散 → Nernst-Einstein 得电导率 → Arrhenius 外推 300K。""")

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_motivation_run.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK:", out, "| cells:", len(cells))
