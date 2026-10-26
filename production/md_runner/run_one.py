"""Run one task of a task map (definitive classification, batch 1 and later).

Generalised from the single-node GPU-pool runner of the replicate campaign:
the scientific protocol of each MD task (temperature, production and
equilibration steps, seed) is read from task_map.json and checked against the
per-task protocol payload and protocol_sha256 in manifest.json; nothing about
the run length is hard-coded. V7's identity checks (model, md.py, source CIF,
package/CUDA/GPU versions) and its raw QC (frame count, step clock, finite
values, frame/log temperature and energy agreement, trajectory sha256) are kept.

Task types (task.sh picks the environment for each):
  md        MACE MD inside the container with macepkg (GPU)
  envcheck  analysis runtime check: Python, exact pins, imports (CPU)
  analyze   run an analysis entry script with the analysis runtime (CPU)

  python3 run_one.py --index N   # inside a job; TASK_MAP env selects the map
  python3 run_one.py --summary   # login node (Python 3.6 compatible): status table
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

import initialization_guard

HERE = Path(__file__).resolve().parent
MODEL_SHA = '2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736'
MD_SHA = 'a773248ae302b8fec18c15cbf5caf78c3e7d21efc687c74ce94576eecc8aa0be'
VERSIONS = {'mace-torch': '0.3.16', 'torch': '2.5.0', 'ase': '3.29.0', 'numpy': '2.2.5'}
# Fixed by the protocol for every MD task; per-task values come from the manifest.
FIXED = {'mlip': 'MACE-MP-0 small, un-fine-tuned', 'model_sha256': MODEL_SHA, 'md_code_sha256': MD_SHA,
         'dtype': 'float32', 'timestep_fs': 1.0, 'friction_ase_inverse_time': 0.01, 'log_every_steps': 50}
DONE = {'md': 'COMPLETED_RAW_VALIDATED', 'envcheck': 'COMPLETED_ENV_VERIFIED', 'analyze': 'COMPLETED'}
ANALYSIS_MODULES = ['numpy', 'scipy', 'matplotlib', 'ase', 'pymatgen.core', 'pymatgen.analysis.diffusion',
                    'kinisi', 'kinisi.analyze', 'scipp', 'emcee']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def task_map_name():
    return os.environ.get('TASK_MAP') or ('smoke_task_map.json' if os.environ.get('SMOKE', '0') == '1' else 'task_map.json')


def output_dir(task):
    if task['task_type'] == 'md':
        return HERE / task['output_root'] / task['candidate_key'] / '{}K_seed{}'.format(task['temperature_K'], task['md_seed'])
    return HERE / task['output_root'] / task['task_id'].replace('smoke/', '')


def load_task(index):
    tasks = read(HERE / task_map_name())
    require(0 <= index < len(tasks) and tasks[index]['index'] == index, 'Invalid task index')
    return tasks[index]


def md_protocol(task):
    """Cross-check task map, manifest payload, protocol sha and the fixed protocol."""
    m = read(HERE / 'manifest.json')
    entries = [t for t in m['tasks'] + m['smoke_tasks'] if t['task_id'] == task['task_id']]
    require(len(entries) == 1, 'Task not registered exactly once in manifest')
    entry = entries[0]
    p = entry['protocol_payload']
    require(object_sha(p) == entry['protocol_sha256'] == task['protocol_sha256'], 'Protocol sha differs (task map / manifest / payload)')
    require(all(p[k] == v for k, v in FIXED.items()), 'Fixed protocol (model/code/dtype/timestep/friction/log) differs')
    for k_task, k_p in (('temperature_K', 'temperature_K'), ('md_seed', 'seed'), ('production_steps', 'production_steps'),
                        ('equilibration_steps', 'equilibration_steps'), ('candidate_key', 'candidate_key')):
        require(task[k_task] == p[k_p] == entry[k_task], 'Task map and manifest differ: ' + k_task)
    require(p['production_steps'] % p['log_every_steps'] == 0 and p['production_steps'] > 0, 'Production steps not a multiple of the log interval')
    c = next(c for c in m['candidates'] if c['key'] == task['candidate_key'])
    require((c['structure_sha256'], c['n_atoms'], c['n_Li'], c['supercell_repetitions']) ==
            (p['structure_sha256'], p['n_atoms'], p['n_Li'], p['supercell_repetitions']), 'Candidate differs from protocol payload')
    require(c.get('initial_atom_order_and_geometry_sha256') == p.get('initial_atom_order_and_geometry_sha256'), 'Canonical initialization differs from protocol payload')
    return p, c


def model_runtime():
    import importlib.metadata
    import torch
    model_path = Path(os.environ['P2_MODEL_PATH'])
    require(torch.cuda.is_available(), 'Allocated GPU not visible')
    require(sha(model_path) == MODEL_SHA, 'MACE checkpoint checksum mismatch')
    require(sha(HERE / 'code/src/md.py') == MD_SHA, 'MD implementation changed')
    runtime = {'host': socket.gethostname(), 'job_id': os.environ.get('PBS_JOBID'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'gpu': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(), 'cuda': torch.version.cuda,
               'model_sha256': MODEL_SHA, 'model_file': 'li7ps6_20260910_seed1/model/mace_mp0_small.model', 'md_sha256': MD_SHA,
               'packages': {k: importlib.metadata.version(k) for k in VERSIONS}, 'python': sys.version.split()[0]}
    require(runtime['packages'] == VERSIONS, 'Package versions differ from baseline')
    require(runtime['cuda'] == '12.4' and runtime['gpu'] == 'NVIDIA A100-SXM4-40GB', 'CUDA/GPU differs from baseline')
    require(runtime['gpu_count'] == 1, 'Expected exactly one visible GPU')
    sys.path.insert(0, str(HERE / 'code'))
    from src import md
    calc = md.load_calculator('mace', device='cuda', mace_model=str(model_path), dtype='float32')
    return md, calc, runtime


def atoms_for(candidate, calc):
    import numpy as np
    from ase.io import read as ase_read
    require(sha(HERE / candidate['structure']) == candidate['structure_sha256'], 'Source CIF bytes changed')
    atoms = ase_read(HERE / candidate['structure'])
    proof = None
    if candidate.get('canonical_initialization_path'):
        # Replicates of earlier seeds (terminal_A1): the V7 canonical initialization and guard.
        require(sha(HERE / candidate['canonical_initialization_path']) == candidate['canonical_initialization_file_sha256'], 'Canonical initialization file changed')
        proof = initialization_guard.validate_initialization(atoms, read(HERE / candidate['canonical_initialization_path']), candidate['initial_atom_order_and_geometry_sha256'])
    require(len(atoms) == candidate['n_atoms'] and int(sum(atoms.numbers == 3)) == candidate['n_Li'], 'Atom count differs')
    supported = {int(z) for z in calc.models[0].atomic_numbers.tolist()}
    require(set(atoms.numbers.tolist()) <= supported, 'Unsupported atomic numbers')
    geometry = {'numbers': atoms.numbers.tolist(), 'positions_A': atoms.positions.tolist(), 'cell_A': atoms.cell.array.tolist(), 'pbc': atoms.pbc.tolist()}
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    require(np.isfinite(energy) and np.isfinite(forces).all(), 'Nonfinite initial energy/force')
    return atoms, {'initial_geometry_sha256': object_sha(geometry), 'initial_energy_per_atom_eV': float(energy) / len(atoms),
                   'initial_max_force_eV_A': float(np.linalg.norm(forces, axis=1).max()), 'initialization_identity_proof': proof}


def validate_raw(task, atoms, series, trajectory, n_steps, equil, log_every):
    """V7 raw QC with the step clock derived from this task's (n_steps, equil, log_every)."""
    import numpy as np
    from ase.io.trajectory import Trajectory
    frames_expected = n_steps // log_every
    expected = np.arange(equil + log_every, equil + n_steps + 1, log_every)
    steps = np.asarray(series['step'])
    require(np.array_equal(steps, np.r_[equil, expected]), 'Log clock or initial sample differs')
    for field in ('T', 'E_per_atom'):
        require(len(series[field]) == frames_expected + 1 and np.isfinite(series[field]).all(), 'Nonfinite or missing log samples')
    with Trajectory(str(trajectory), 'r') as frames:
        require(len(frames) == frames_expected, 'Full {} production frames required'.format(frames_expected))
        for i, frame in enumerate(frames):
            require(np.array_equal(frame.numbers, atoms.numbers) and np.array_equal(frame.pbc, atoms.pbc) and np.allclose(frame.cell.array, atoms.cell.array, atol=1e-10, rtol=0), 'Frame order/cell differs')
            for value in (frame.positions, frame.get_velocities(), frame.get_forces(), frame.get_potential_energy(), frame.get_temperature()):
                require(value is not None and np.isfinite(value).all(), 'Nonfinite/missing frame data')
            require(np.isclose(frame.get_potential_energy() / len(frame), series['E_per_atom'][i + 1], atol=1e-8, rtol=1e-8) and np.isclose(frame.get_temperature(), series['T'][i + 1], atol=1e-6, rtol=1e-8), 'Frame/log energy or temperature mismatch')
    mean_T = float(np.mean(series['T'][1:]))
    return {'passed': True, 'production_frames': frames_expected, 'log_samples': frames_expected + 1, 'frame_dt_ps': log_every / 1000,
            'first_production_step': int(expected[0]), 'last_production_step': int(expected[-1]), 'mean_T_production_K': mean_T,
            'mean_T_relative_deviation': abs(mean_T - task['temperature_K']) / task['temperature_K'],
            'trajectory_sha256': sha(trajectory), 'task': task}


def start(task, index, skip_done=True):
    """Skip completed work; move an earlier attempt aside; return (out, status) or None."""
    out = output_dir(task)
    if skip_done and (out / 'status.json').is_file() and read(out / 'status.json').get('status') == DONE[task['task_type']]:
        print(json.dumps({'index': index, 'status': 'SKIPPED_ALREADY_DONE', 'task_id': task['task_id']}))
        return None
    if out.exists():
        # Keep the failed/interrupted attempt as evidence; never write into it again.
        attempt = out.with_name('{}.attempt_{}'.format(out.name, dt.datetime.now().strftime('%Y%m%dT%H%M%S')))
        shutil.move(str(out), str(attempt))
        print('Previous incomplete attempt moved to {}'.format(attempt.relative_to(HERE)))
    out.mkdir(parents=True)
    status = {'status': 'RUNNING', 'task': task, 'task_map': task_map_name(), 'job_id': os.environ.get('PBS_JOBID'),
              'started_utc': now(), 'campaign': HERE.name, 'smoke': task['output_root'] != 'results'}
    write(out / 'status.json', status)
    return out, status


def run_md(task, index):
    p, c = md_protocol(task)
    n_steps, equil, log_every = p['production_steps'], p['equilibration_steps'], p['log_every_steps']
    started = start(task, index)
    if started is None:
        return
    out, status = started
    status.update(n_steps=n_steps, equilibration_steps=equil, protocol_sha256=task['protocol_sha256'])
    write(out / 'status.json', status)
    try:
        md, calc, runtime = model_runtime()
        atoms, initial = atoms_for(c, calc)
        runtime.update(initial)
        write(out / 'runtime.json', runtime)
        trajectory = out / 'md.traj'
        summary = md.run_nvt(atoms, calc, temperature_K=p['temperature_K'], n_steps=n_steps, timestep_fs=p['timestep_fs'],
                             friction=p['friction_ase_inverse_time'], equilib_steps=equil, traj_path=str(trajectory),
                             log_every=log_every, seed=p['seed'])
        series = summary.pop('series')
        write(out / 'series.json', series)
        qc = validate_raw(task, atoms, series, trajectory, n_steps, equil, log_every)
        write(out / 'raw_qc.json', qc)
        steps_per_s = (n_steps + equil) / summary['wall_seconds']
        summary.update(traj_path='md.traj', task=task, seed=p['seed'], n_frames=qc['production_frames'], n_logged_samples=qc['log_samples'],
                       equilib_steps=equil, model_sha256=MODEL_SHA, structure_sha256=c['structure_sha256'],
                       protocol_sha256=task['protocol_sha256'], protocol_payload=p, steps_per_s=steps_per_s)
        write(out / 'md_run.json', summary)
        status.update(status=DONE['md'], completed_utc=now(), production_frames=qc['production_frames'], wall_seconds=summary['wall_seconds'],
                      steps_per_s=steps_per_s, mean_T=summary['mean_T'],
                      outputs_sha256={n: sha(out / n) for n in ('md.traj', 'series.json', 'raw_qc.json', 'runtime.json', 'md_run.json')})
        write(out / 'status.json', status)
        print(json.dumps({'index': index, 'status': DONE['md'], 'task_id': task['task_id'], 'wall_h': round(summary['wall_seconds'] / 3600, 3),
                          'steps_per_s': round(steps_per_s, 2), 'mean_T': summary['mean_T']}))
    except BaseException as exc:
        status.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), completed_utc=now())
        write(out / 'status.json', status)
        raise


def lock_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, version = line.split('==')
            pins[name.strip()] = version.strip()
    return pins


def run_envcheck(task, index):
    started = start(task, index)
    if started is None:
        return
    out, status = started
    try:
        import importlib
        import importlib.metadata
        import platform
        lock = HERE / task['requirements_lock']
        require(sha(lock) == task['requirements_lock_sha256'], 'Requirements lock changed')
        pins = lock_pins(lock)
        imported = {name: getattr(importlib.import_module(name), '__file__', '') for name in ANALYSIS_MODULES}
        versions = {name: importlib.metadata.version(name) for name in pins}
        prefix = Path(sys.prefix).resolve()
        outside = {k: v for k, v in imported.items() if v and not Path(v).resolve().is_relative_to(prefix)}
        # Functional touch: parse one census supercell CIF with ASE and pymatgen.
        from ase.io import read as ase_read
        from pymatgen.core import Structure
        cif = HERE / 'structures' / 'p1r45_mp755309.cif'
        n_ase, n_pmg = len(ase_read(cif)), len(Structure.from_file(str(cif)))
        report = {'python': sys.version.split()[0], 'implementation': platform.python_implementation(), 'platform': platform.platform(),
                  'executable_in_runtime': str(Path(sys.executable).resolve()).startswith('/tmp/'), 'pins': pins, 'versions': versions,
                  'pins_match': versions == pins, 'imported_modules': sorted(imported), 'modules_outside_runtime': outside,
                  'cif_atoms_ase': n_ase, 'cif_atoms_pymatgen': n_pmg, 'host': socket.gethostname(),
                  'in_container': Path('/.singularity.d').exists()}
        write(out / 'envcheck.json', report)
        require(report['python'] == task['python_version'], 'Python version differs from the validated analysis environment')
        require(report['pins_match'], 'Installed versions differ from requirements-analysis.lock.txt')
        require(not outside, 'Analysis modules imported from outside the runtime')
        require(n_ase == n_pmg == 216, 'CIF parse differs')
        status.update(status=DONE['envcheck'], completed_utc=now(), envcheck_sha256=sha(out / 'envcheck.json'))
        write(out / 'status.json', status)
        print(json.dumps({'index': index, 'status': DONE['envcheck'], 'python': report['python'], 'versions': versions}))
    except BaseException as exc:
        status.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), completed_utc=now())
        write(out / 'status.json', status)
        raise


def run_analyze(task, index):
    """Run task['entry'] (campaigns/p2_definitive_v1/analysis/analyze_campaign.py) with the analysis runtime.

    Always re-executes (the analysis reuses finished runs by trajectory/code/config sha),
    so the same task can be resubmitted after more MD has completed; the previous
    status folder is moved aside."""
    entry = HERE / task['entry']
    require(entry.is_file(), 'Analysis entry missing: ' + task['entry'])
    for path, expected in task.get('code_sha256', {}).items():
        require(sha(HERE / path) == expected, 'Analysis code changed: ' + path)
    out, status = start(task, index, skip_done=False)
    args = [str(a).replace('{out}', str(out)).replace('{campaign}', str(HERE)) for a in task.get('args', [])]
    status.update(args=[a.replace(str(HERE), '{campaign}') for a in args])
    write(out / 'status.json', status)
    with (out / 'analysis.log').open('w') as log:
        rc = subprocess.call([sys.executable, '-E', '-s', '-u', '-B', str(entry)] + args, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)
    tables = {}
    for a, b in zip(task['args'], task['args'][1:]):
        if a == '--out':
            dest = Path(b.replace('{campaign}', str(HERE)))
            tables = {f.name: sha(f) for f in sorted(dest.glob('*')) if f.is_file()}
    status.update(status=DONE['analyze'] if rc == 0 else 'FAILED', returncode=rc, completed_utc=now(), analysis_tables_sha256=tables,
                  outputs_sha256={f.relative_to(out).as_posix(): sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'status.json'})
    write(out / 'status.json', status)
    require(rc == 0, 'Analysis entry failed with rc={}'.format(rc))


def summary():
    tasks = read(HERE / task_map_name())
    counts = {}
    for t in tasks:
        f = output_dir(t) / 'status.json'
        s = read(f) if f.is_file() else {'status': 'NOT_STARTED'}
        counts[s['status']] = counts.get(s['status'], 0) + 1
        if s.get('wall_seconds'):
            extra = '{:.2f} h  {:.1f} steps/s'.format(s['wall_seconds'] / 3600, s.get('steps_per_s', 0))
        else:
            extra = s.get('error', s.get('started_utc', ''))
        print('{:>2}  {:<9} {:<40} {:<24} {}'.format(t['index'], t['task_type'], t['task_id'], s['status'], extra))
    print(json.dumps(counts))


if __name__ == '__main__':
    if sys.argv[1:] == ['--summary']:
        summary()
    elif len(sys.argv) == 3 and sys.argv[1] == '--index':
        idx = int(sys.argv[2])
        current = load_task(idx)
        {'md': run_md, 'envcheck': run_envcheck, 'analyze': run_analyze}[current['task_type']](current, idx)
    else:
        sys.exit('usage: run_one.py --index N | --summary   (TASK_MAP or SMOKE=1 selects the task map)')
