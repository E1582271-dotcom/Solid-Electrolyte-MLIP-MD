"""Source-bound canonical geometry comparison; retain native ASE arrays unchanged."""
import hashlib
import json
ABSOLUTE_TOLERANCE_A=1e-12

def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(value,message):
    if not value:raise RuntimeError(message)

def validate_initialization(atoms,reference,expected_sha):
    import numpy as np
    require(isinstance(reference,dict) and set(reference)=={'numbers','positions_A','cell_A','pbc'},'Canonical initialization schema differs')
    require(object_sha(reference)==expected_sha,'Canonical initialization reference checksum differs')
    positions=np.asarray(reference['positions_A'],dtype=np.float64);cell=np.asarray(reference['cell_A'],dtype=np.float64)
    numbers=np.asarray(reference['numbers']);pbc=np.asarray(reference['pbc'])
    require(numbers.shape==(len(atoms),) and positions.shape==atoms.positions.shape==(len(atoms),3) and cell.shape==atoms.cell.array.shape==(3,3) and pbc.shape==atoms.pbc.shape==(3,),'Initialization shape/count differs')
    require(np.array_equal(atoms.numbers,numbers),'Initial species/atom order differs')
    require(np.array_equal(atoms.pbc,pbc),'Initial PBC differs')
    require(all(np.isfinite(a).all() for a in (positions,cell,atoms.positions,atoms.cell.array)),'Nonfinite initialization geometry')
    position_delta=float(np.max(np.abs(atoms.positions-positions)));cell_delta=float(np.max(np.abs(atoms.cell.array-cell)))
    require(position_delta<=ABSOLUTE_TOLERANCE_A and cell_delta<=ABSOLUTE_TOLERANCE_A,'Initial coordinates/cell exceed source-proven float64 tolerance')
    native={'numbers':atoms.numbers.tolist(),'positions_A':atoms.positions.tolist(),'cell_A':atoms.cell.array.tolist(),'pbc':atoms.pbc.tolist()}
    return {'passed':True,'canonical_initialization_sha256':expected_sha,'actual_initialization_sha256':object_sha(native),'max_position_delta_A':position_delta,'max_cell_delta_A':cell_delta,'absolute_tolerance_A':ABSOLUTE_TOLERANCE_A,'relative_tolerance':0,'native_arrays_unchanged':True,'identity_definition':'Exact source CIF bytes, canonical reference hash, species/order/PBC; finite native Cartesian/cell absolute tolerance only; no coordinate rounding or replacement'}
