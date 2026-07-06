"""Generates the merged 00_motivation_run.ipynb: install packages -> LiCl single-point
energy -> 600K MD -> relaxation -> real Li6PS5Cl single-point energy.
Written via json to guarantee valid UTF-8 JSON and avoid copy-paste mangling. Upload the
one resulting file to Colab and Run all."""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# P0 motivation run -- your first ML-potential workflow (one file, end to end)

Runs four things back to back to build "this actually works" confidence + touch every
basic skill needed later for the conductivity calculation:
1. **Single-point energy**: compute a crystal's energy with the pretrained universal
   potential MACE-MP
2. **Molecular dynamics (MD)**: 600K Langevin thermostat, watch the temperature settle
   and the energy stay bounded
3. **Structure relaxation**: rattle a crystal, watch the MLIP push it back down to a
   low-energy configuration along the force
4. **A real system**: pull real Li6PS5Cl from Materials Project and compute its
   single-point energy -- the entry point into flagship Project 2

**Usage**: Colab -> `Runtime` -> `Change runtime type` -> **T4 GPU** -> upload this file ->
`Runtime` -> **Run all**.
**Prerequisite**: create `MP_API_KEY` under the left-hand \U0001F511 Secrets panel (value = your MP
key) and grant this notebook access to it.
**Hard-won lessons**: ① the runtime must be switched to GPU (cell 1 needs `CUDA: True`);
② use float32 for the hello-world, not float64 (a T4's double-precision throughput is only
1/32 of its single-precision throughput).""")

code("""# 1) Install packages + confirm GPU (~1-2 minutes)
!pip install -q mace-torch mp-api ase pymatgen
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print('CUDA:', torch.cuda.is_available(), '| GPU:', gpu, '| device:', device)
# If this prints False/CPU: Change runtime type -> T4 GPU -> Run all""")

code("""# 2) Build a small crystal + attach the MACE-MP universal potential, compute a single-point energy
from ase.build import bulk
from mace.calculators import mace_mp

calc = mace_mp(model='small', dispersion=False, default_dtype='float32', device=device)  # this calculator is reused below
atoms = bulk('LiCl', 'rocksalt', a=5.14) * (3, 3, 3)   # 3x3x3 supercell = 216 atoms
atoms.calc = calc
print(len(atoms), 'atoms')
print('Potential energy =', round(atoms.get_potential_energy(), 3), 'eV')   # one line, MACE-MP gives the energy""")

md("""## (2) Molecular dynamics: run 2 ps at 600K, watch the thermostat settle the temperature

- The temperature will initially be **halved** (equipartition: kinetic energy splits half
  into potential energy); the Langevin thermostat then slowly warms it back to 600K.
- Large temperature oscillations are **real physics** (finite-size thermal fluctuation,
  sigma_T/T = sqrt(2/3N)), not noise -- the larger the system, the smaller the swing.""")

code("""# 3) 600K NVT (Langevin) MD, 2000 steps x 1 fs = 2 ps, print progress every 200 steps
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
print('Done, took', round(time.time()-t0,1), 'seconds')""")

code("""# 4) Plot: temperature should oscillate strongly around 600K, energy per atom should stay stable, no blow-up
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(steps, T); ax[0].axhline(600, ls='--', c='r'); ax[0].set(xlabel='step', ylabel='T (K)', title='Temperature')
ax[1].plot(steps, E); ax[1].set(xlabel='step', ylabel='E/atom (eV)', title='Potential energy')
plt.tight_layout(); plt.show()""")

md("""## (3) Structure relaxation: rattle -> let the MLIP push it back down along the force

Relaxation = move the atoms along the force (force = -grad E) to a local energy minimum.
The other half of the basic skill set besides MD.""")

code("""# 5) Relaxation: deliberately rattle a crystal, then let the MLIP push it back to a low-energy configuration
from ase.optimize import BFGS

relax_atoms = bulk('LiCl', 'rocksalt', a=5.14) * (2, 2, 2)
relax_atoms.calc = calc                        # reuse the same MACE-MP calculator
E0 = relax_atoms.get_potential_energy()
relax_atoms.rattle(stdev=0.15, seed=0)         # add a ~0.15 A random displacement to each atom
E_rattled = relax_atoms.get_potential_energy()
BFGS(relax_atoms, logfile=None).run(fmax=0.02, steps=100)   # relax until the max force < 0.02 eV/A
E_relaxed = relax_atoms.get_potential_energy()

print(f'Ideal crystal   E0        = {E0:.3f} eV')
print(f'Rattled         E_rattled = {E_rattled:.3f} eV   (raised by {E_rattled - E0:+.3f})')
print(f'Relaxed         E_relaxed = {E_relaxed:.3f} eV   (falls back near E0 = the MLIP pushed the structure back)')
print(f'Relaxation converged, fmax = {abs(relax_atoms.get_forces()).max():.4f} eV/A')""")

md("""## (4) A real system: pull Li6PS5Cl from Materials Project, compute its single-point energy

This formally connects to flagship Project 2. Li6PS5Cl is a disordered structure (Cl/S
disordered over the 4a/4c sites); today we start with the most stable **ordered
approximant**, and handle the disorder properly starting W5.""")

code("""# 6) Pull the real Li6PS5Cl structure from Materials Project
import os
from mp_api.client import MPRester

try:
    from google.colab import userdata
    MP_API_KEY = userdata.get('MP_API_KEY')          # read from a Colab Secret, never hard-coded
except Exception:
    MP_API_KEY = os.environ.get('MP_API_KEY')
assert MP_API_KEY, "MP_API_KEY not found: check that MP_API_KEY is set under the left-hand Secrets panel and that this notebook has access enabled"

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        formula="Li6PS5Cl",
        fields=["material_id", "formula_pretty", "energy_above_hull", "symmetry", "structure"])

print(f"Found {len(docs)} Li6PS5Cl entries")
ordered = [d for d in docs if d.energy_above_hull is not None and d.structure.is_ordered]
print(f"{len(ordered)} of them are ordered (directly runnable):")
for d in sorted(ordered, key=lambda x: x.energy_above_hull):
    sg = d.symmetry.symbol if d.symmetry else '?'
    print(f"  {d.material_id:12s}  E_hull={d.energy_above_hull:.3f} eV/atom  {sg}  {len(d.structure)} atoms")

if ordered:
    best = min(ordered, key=lambda x: x.energy_above_hull)
    struct = best.structure
    print()
    print(f"-> Selected the most stable ordered approximant: {best.material_id}  ({len(struct)} atoms/cell)")
else:
    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
    valid = [d for d in docs if d.energy_above_hull is not None]
    best = min(valid, key=lambda x: x.energy_above_hull)
    print()
    print(f"All entries are disordered; ordering the most stable one, {best.material_id}...")
    struct = OrderDisorderedStructureTransformation().apply_transformation(best.structure)
    print(f"-> Ordering complete: {len(struct)} atoms/cell")""")

code("""# 7) Compute a MACE-MP single-point energy for the real Li6PS5Cl (same machinery as LiCl above, just a different, flagship, system)
real = struct.to_ase_atoms()
real.calc = calc                               # reuse the same MACE-MP calculator
e = real.get_potential_energy()
print("Formula =", real.get_chemical_formula(), "| atoms =", len(real))
print(f"MACE-MP single-point energy = {e:.3f} eV  ({e/len(real):.4f} eV/atom)")
print(f"Max force fmax  = {abs(real.get_forces()).max():.3f} eV/A  (unrelaxed, nonzero is expected)")""")

md("""## Everything ran = environment ready + all four basic skills in hand

You can now: **compute a single-point energy / run MD / relax a structure / pull a real
structure from a database and feed it to an ML potential**. This is the exact same
machinery the capstone uses to compute Li+ conductivity.

**Next (W5)**: formally handle Li6PS5Cl's disorder -- enumerate configurations,
multi-temperature MD, compute Li+ diffusion from the trajectories -> Nernst-Einstein for
conductivity -> Arrhenius extrapolation to 300K.""")

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_motivation_run.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK:", out, "| cells:", len(cells))
