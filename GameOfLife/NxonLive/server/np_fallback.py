# Multi Neuraxon Game of Life 5 — minimal pure-Python numpy fallback  [v189-compat substrate]
# Based on the Paper:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# ===================================================================
# PyPy cannot load CPython's C-extension numpy. The Neuraxon substrate
# only uses a TINY numpy surface:
#   np.full((10,10,4), v) · np.zeros_like(a) · np.array(nested) ·
#   a.tolist() · 2D/3D strided slicing+assignment for the 10x10x4
#   neuromodulator-diffusion Laplacian · np.mean / np.var
# This module implements exactly that, with strided VIEWS so an
# in-place `grid[:, :, i] += ...` writes back through the slice
# (required for the diffusion to be correct).
#
# It is installed into sys.modules['numpy'] ONLY when the real numpy
# fails to import (see server/np_fallback.install()). On CPython with
# a working numpy nothing changes; on PyPy this makes the substrate
# run with zero numpy dependency.
# ===================================================================
import math as _math


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class ndarray:
    __slots__ = ("_d", "shape", "_off", "_st")

    def __init__(self, data, shape, off=0, strides=None):
        self._d = data                       # shared flat list
        self.shape = tuple(shape)
        self._off = off
        if strides is None:
            strides = []
            acc = 1
            for s in reversed(shape):
                strides.append(acc)
                acc *= s
            strides.reverse()
        self._st = tuple(strides)

    # ---- indexing -------------------------------------------------
    def _norm(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # pad with full slices
        key = list(key) + [slice(None)] * (len(self.shape) - len(key))
        new_shape, new_strides, off = [], [], self._off
        for ax, k in enumerate(key):
            n = self.shape[ax]
            st = self._st[ax]
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                cnt = max(0, (stop - start + (step - (1 if step > 0
                          else -1))) // step)
                off += start * st
                new_shape.append(cnt)
                new_strides.append(st * step)
            else:
                if k < 0:
                    k += n
                off += k * st
        return new_shape, new_strides, off

    def __getitem__(self, key):
        sh, st, off = self._norm(key)
        if not sh:                            # scalar
            return self._d[off]
        return ndarray(self._d, sh, off, st)

    def _iter_index(self):
        # yield flat offsets in row-major order for this view
        if not self.shape:
            yield self._off
            return
        idx = [0] * len(self.shape)
        while True:
            o = self._off
            for a, i in enumerate(idx):
                o += i * self._st[a]
            yield o
            ax = len(self.shape) - 1
            while ax >= 0:
                idx[ax] += 1
                if idx[ax] < self.shape[ax]:
                    break
                idx[ax] = 0
                ax -= 1
            if ax < 0:
                return

    def __setitem__(self, key, val):
        sh, st, off = self._norm(key)
        tgt = ndarray(self._d, sh, off, st)
        offs = list(tgt._iter_index())
        if isinstance(val, ndarray):
            vs = [val._d[o] for o in val._iter_index()]
            for o, v in zip(offs, vs):
                tgt._d[o] = v
        else:
            for o in offs:
                tgt._d[o] = val

    # ---- elementwise ops -----------------------------------------
    def _binop(self, other, f):
        offs = list(self._iter_index())
        out = [0.0] * len(offs)
        if isinstance(other, ndarray):
            oo = list(other._iter_index())
            for k, (a, b) in enumerate(zip(offs, oo)):
                out[k] = f(self._d[a], other._d[b])
        else:
            for k, a in enumerate(offs):
                out[k] = f(self._d[a], other)
        return ndarray(out, self.shape)

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o): return self._binop(o, lambda a, b: b + a)
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._binop(o, lambda a, b: b - a)
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._binop(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._binop(o, lambda a, b: a / b)

    def __iadd__(self, o):
        res = self.__add__(o)
        self.__setitem__((slice(None),) * len(self.shape), res)
        return self

    # ---- conversion ----------------------------------------------
    def tolist(self):
        if len(self.shape) == 1:
            return [self._d[o] for o in self._iter_index()]
        return [self[i].tolist() for i in range(self.shape[0])]

    def __len__(self):
        return self.shape[0] if self.shape else 0


# ---- module-level constructors / reductions ----------------------
def _flatten(seq, out):
    if isinstance(seq, (list, tuple)):
        for s in seq:
            _flatten(s, out)
    else:
        out.append(seq)


def _shape_of(seq):
    sh = []
    s = seq
    while isinstance(s, (list, tuple)):
        sh.append(len(s))
        s = s[0] if s else None
    return tuple(sh)


def array(seq, dtype=None):
    if isinstance(seq, ndarray):
        return seq
    flat = []
    _flatten(seq, flat)
    return ndarray([float(x) for x in flat], _shape_of(seq))


def full(shape, val, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    return ndarray([float(val)] * _prod(shape), shape)


def zeros_like(a):
    if isinstance(a, ndarray):
        return ndarray([0.0] * _prod(a.shape), a.shape)
    return full(_shape_of(a), 0.0)


def zeros(shape, dtype=None):
    return full(shape, 0.0)


def _values(x):
    if isinstance(x, ndarray):
        return [x._d[o] for o in x._iter_index()]
    out = []
    _flatten(x, out)
    return [float(v) for v in out]


def mean(x):
    v = _values(x)
    return sum(v) / len(v) if v else 0.0


def var(x):
    v = _values(x)
    if not v:
        return 0.0
    m = sum(v) / len(v)
    return sum((t - m) ** 2 for t in v) / len(v)


def std(x):
    return _math.sqrt(var(x))


def sqrt(x):
    return _math.sqrt(x)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# numpy attribute shims occasionally referenced
float64 = float
float32 = float
int64 = int
ndarray_type = ndarray
pi = _math.pi


IS_FALLBACK = True            # consumers can detect the pure-Python shim


def install():
    """Make `import numpy` work everywhere in THIS process.

    If a real, importable numpy exists (normal CPython) we do nothing
    and the substrate uses it unchanged. If it is missing or broken
    (PyPy loading CPython's C-extension build → ModuleNotFoundError:
    numpy.core._multiarray_umath), we register THIS module as
    `numpy` in sys.modules BEFORE the substrate imports it, so the
    Neuraxon network runs with zero numpy dependency.

    Must be called at the very top of every process entrypoint
    (engine import, each brain-pool worker, the web server) before
    any `neuraxon` import.
    """
    import sys
    if "numpy" in sys.modules and getattr(
            sys.modules["numpy"], "__name__", "") != __name__:
        return  # a real numpy is already imported and working
    try:
        import numpy  # noqa: F401
        # force the C-extension to actually load; PyPy raises here
        numpy.zeros(1)
        return
    except Exception:
        pass
    sys.modules["numpy"] = sys.modules[__name__]

