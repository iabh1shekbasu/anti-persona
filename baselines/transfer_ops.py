"""The baselines' own input transformations and kernels, taken from the authors' repositories.

The authors' functions are compiled by AST from their pinned source files, never imported (their modules load
surrogate models / other packages at import time). Clone the repositories as described in third_party/README.md, or
point the environment variables at existing clones:
  TRANSFERATTACK_DIR  Trustworthy-AI-Group/TransferAttack @ ae960f5   (SIA block transforms, TIM kernel)
  GLEAM_DIR           LuckAlex/GLEAM @ 2ec57f3                       (GLEAM-v LFE / GDE warps, TI kernel)
"""
import ast
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIRD_PARTY = Path(__file__).resolve().parent / 'third_party'
TRANSFERATTACK_DIR = Path(os.environ.get('TRANSFERATTACK_DIR', THIRD_PARTY / 'TransferAttack'))
GLEAM_DIR = Path(os.environ.get('GLEAM_DIR', THIRD_PARTY / 'GLEAM'))
SIA_FILE = TRANSFERATTACK_DIR / 'transferattack/input_transformation/sia.py'
TIM_FILE = TRANSFERATTACK_DIR / 'transferattack/input_transformation/tim.py'
NURBS_FILE = GLEAM_DIR / 'NURBSAttacker.py'
GLEAM_UTILS_FILE = GLEAM_DIR / 'utils.py'
SIA_COMMIT = 'ae960f546a907a193931a2df12afd0f3fef2e0a5'
GLEAM_COMMIT = '2ec57f387c0ae53d74e69d2cbaecc820589e933a'

SIA_METHODS = ('vertical_shift', 'horizontal_shift', 'vertical_flip', 'horizontal_flip', 'rotate180', 'scale',
               'add_noise', 'blocktransform', 'transform')
SIA_OPS = ['vertical_shift', 'horizontal_shift', 'vertical_flip', 'horizontal_flip', 'rotate180', 'scale', 'add_noise']
SIA_DEFAULTS = dict(num_scale=20, num_block=3, decay=1.)
TIM = dict(kernel_type='gaussian', kernel_size=15, nsig=3)
# GLEAM-v: eval_gleam.py (num_warping 20, mesh 30 x 30), dem_trans U(1.1, 1.8), paper Sec. 4.1.4 (displacement 10 px),
# ImageAttacker kernel_size 5
GLEAM = dict(num_warping=20, mesh=30, warp_offset=10.0, scale_low=1.1, scale_high=1.8, ti_kernel=5, decay=1.0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _need(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{path} not found: clone the third-party repository (baselines/third_party/README.md) '
                                'or set TRANSFERATTACK_DIR / GLEAM_DIR')


def ast_functions(path, names, env, cls=None):
    """Compile exactly the named functions (top level, or methods of class `cls`) of a source file into a fresh
    namespace. The file's module is never imported, so its imports and module-level code never run."""
    _need(path)
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    body = tree.body
    if cls is not None:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        assert len(classes) == 1, (path, cls)
        body = classes[0].body
    defs = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            defs.setdefault(node.name, []).append(node)
    found = {}
    for name in names:
        assert len(defs.get(name, [])) == 1, (path, name, len(defs.get(name, [])))
        found[name] = defs[name][0]
    namespace = dict(env)
    exec(compile(ast.Module(body=list(found.values()), type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def sia_init_contract(path=SIA_FILE):
    """The SIA constructor's defaults and op list, read from the source (the constructor loads a surrogate, so it is
    never called)."""
    _need(path)
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SIA')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    names = [a.arg for a in init.args.args]
    defaults = dict(zip(names[len(names) - len(init.args.defaults):], (ast.literal_eval(d) if not isinstance(d, ast.BinOp)
                    else ast.unparse(d) for d in init.args.defaults)))
    ops = None
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'self.op':
            ops = [ast.unparse(e).removeprefix('self.') for e in node.value.elts]
    return dict(num_scale=defaults['num_scale'], num_block=defaults['num_block'], decay=defaults['decay'], ops=ops)


class SIATransform:
    """TransferAttack SIA's input transformation with its own methods (compiled from sia.py) and its defaults.
    The block ops act on [0, 1] images: view = blocktransform(image / 255) * 255."""

    def __init__(self, path=SIA_FILE):
        contract = sia_init_contract(path)
        assert contract == dict(**SIA_DEFAULTS, ops=SIA_OPS), contract
        methods = ast_functions(path, SIA_METHODS, dict(torch=torch, F=F, np=np), cls='SIA')
        for name, fn in methods.items():
            setattr(self, name, fn.__get__(self))
        self.num_scale, self.num_block = contract['num_scale'], contract['num_block']
        self.op = [getattr(self, name) for name in contract['ops']]
        self.source_sha256 = sha(path)


def tim_kernel(device=None):
    """TransferAttack TIM.generate_kernel('gaussian', 15) (nsig 3), compiled from tim.py; shape [3, 1, 15, 15]."""
    import scipy.stats as st
    fns = ast_functions(TIM_FILE, ('generate_kernel',), dict(np=np, st=st, torch=torch), cls='TIM')
    holder = type('TIMHolder', (), {'device': device or 'cpu'})()
    return fns['generate_kernel'](holder, TIM['kernel_type'], TIM['kernel_size'])


class GleamTransforms:
    """GLEAM-v's LFE (NURBS free-form warp) and GDE (random scale + pad) views and its TI kernel, compiled from the
    GLEAM repository. CMFA (the text branch) is omitted: this task has no attacker-controlled text."""

    def __init__(self):
        import scipy.stats as st
        nurbs = ast_functions(NURBS_FILE, ('bspline_basis', 'init_control_points', 'nurbs_ffd_torch'), dict(torch=torch, F=F))
        assert nurbs['nurbs_ffd_torch'].__globals__['bspline_basis'] is nurbs['bspline_basis']
        self.init_control_points = nurbs['init_control_points']
        self.nurbs_ffd_torch = nurbs['nurbs_ffd_torch']
        self._dim = ast_functions(NURBS_FILE, ('dim',), dict(torch=torch, F=F), cls='ImageAttacker')['dim']
        utils = ast_functions(GLEAM_UTILS_FILE, ('gkern', 'get_kernel'), dict(np=np, st=st, torch=torch))
        self.get_kernel = utils['get_kernel']
        self.source_sha256 = {str(NURBS_FILE): sha(NURBS_FILE), str(GLEAM_UTILS_FILE): sha(GLEAM_UTILS_FILE)}

    def gde(self, image):
        """dem_trans view: dim(image, r), r ~ U(1.1, 1.8) from the CPU global torch RNG."""
        rate = torch.empty(1).uniform_(GLEAM['scale_low'], GLEAM['scale_high']).item()
        return self._dim(None, image, rate)

    def lfe(self, image):
        """vwt_trans view with control-point displacement U(-10, 10) px (paper; the release passes +-1)."""
        control_points, weights = self.init_control_points(image, GLEAM['mesh'], GLEAM['mesh'],
                                                           min_offset=-GLEAM['warp_offset'], max_offset=GLEAM['warp_offset'])
        return self.nurbs_ffd_torch(image, GLEAM['mesh'], GLEAM['mesh'], control_points, weights, image.shape[0])

    def kernel(self, device, dtype):
        return self.get_kernel(GLEAM['ti_kernel']).to(device=device, dtype=dtype)
