import ast
import copy
import dataclasses
import datetime
import decimal
import re
from collections import ChainMap, Counter, UserDict, UserList, UserString, defaultdict, deque, namedtuple, OrderedDict

import sys
import uuid
from enum import Enum, Flag, IntFlag, auto
from pathlib import Path
import array # Add import for array

import pydantic
import pytest

from codeflash.either import Failure, Success
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType
from codeflash.verification.comparator import comparator, _extract_exception_from_message, _get_wrapped_exception
from codeflash.verification.equivalence import compare_test_results


def test_basic_python_objects() -> None:
    a = 5
    b = 5
    c = 6
    d = None
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = 5.0
    b = 5.0
    c = 6.0
    d = None
    e = None
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(d, a)
    assert comparator(d, e)

    a = "Hello"
    b = "Hello"
    c = "World"
    assert comparator(a, b)
    assert not comparator(a, c)

    a = [1, 2, 3]
    b = [1, 2, 3]
    c = [1, 2, 4]
    assert comparator(a, b)
    assert not comparator(a, c)

    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 2}
    c = {"a": 1, "b": 3}
    d = {"c": 1, "b": 2}
    e = {"a": 1, "b": 2, "c": 3}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(a, e)

    a = (1, 2, "str")
    b = (1, 2, "str")
    c = (1, 2, "str2")
    d = [1, 2, "str"]
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = {1, 2, 3}
    b = {2, 3, 1}
    c = {1, 2, 4}
    d = {1, 2, 3, 4}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = (65).to_bytes(1, byteorder="big")
    b = (65).to_bytes(1, byteorder="big")
    c = (66).to_bytes(1, byteorder="big")
    assert comparator(a, b)
    assert not comparator(a, c)
    a = (65).to_bytes(2, byteorder="little")
    b = (65).to_bytes(2, byteorder="big")
    assert not comparator(a, b)

    a = bytearray([65, 64, 63])
    b = bytearray([65, 64, 63])
    c = bytearray([65, 64, 62])
    assert comparator(a, b)
    assert not comparator(a, c)

    memoryview_a = memoryview(bytearray([65, 64, 63]))
    memoryview_b = memoryview(bytearray([65, 64, 63]))
    memoryview_c = memoryview(bytearray([65, 64, 62]))
    assert comparator(memoryview_a, memoryview_b)
    assert not comparator(memoryview_a, memoryview_c)

    a = frozenset([1, 2, 3])
    b = frozenset([2, 3, 1])
    c = frozenset([1, 2, 4])
    d = frozenset([1, 2, 3, 4])
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = map
    b = pow
    c = pow
    d = abs
    assert comparator(b, c)
    assert not comparator(a, b)
    assert not comparator(c, d)

    a = object()
    b = object()
    c = abs
    assert comparator(a, b)
    assert not comparator(a, c)

    a = type([])
    b = type([])
    c = type({})
    assert comparator(a, b)
    assert not comparator(a, c)

@pytest.mark.parametrize("r1, r2, expected", [
    (range(1, 10), range(1, 10), True),                # equal
    (range(0, 10), range(1, 10), False),               # different start
    (range(2, 10), range(1, 10), False),
    (range(1, 5), range(1, 10), False),                # different stop
    (range(1, 20), range(1, 10), False),
    (range(1, 10, 1), range(1, 10, 2), False),          # different step
    (range(1, 10, 3), range(1, 10, 2), False),
    (range(-5, 0), range(-5, 0), True),                # negative ranges
    (range(-10, 0), range(-5, 0), False),
    (range(5, 1), range(10, 5), True),                # empty ranges
    (range(5, 1), range(5, 1), True),
    (range(7), range(0, 7), True),
    (range(0, 7), range(0, 7, 1), True),
    (range(7), range(0, 7, 1), True),
])

def test_ranges(r1, r2, expected):
    assert comparator(r1, r2) == expected


def test_standard_python_library_objects() -> None:
    a = datetime.datetime(2020, 2, 2, 2, 2, 2) # type: ignore
    b = datetime.datetime(2020, 2, 2, 2, 2, 2) # type: ignore
    c = datetime.datetime(2020, 2, 2, 2, 2, 3) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.date(2020, 2, 2) # type: ignore
    b = datetime.date(2020, 2, 2) # type: ignore
    c = datetime.date(2020, 2, 3) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.timedelta(days=1) # type: ignore
    b = datetime.timedelta(days=1) # type: ignore
    c = datetime.timedelta(days=2) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.time(2, 2, 2) # type: ignore
    b = datetime.time(2, 2, 2) # type: ignore
    c = datetime.time(2, 2, 3) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.timezone.utc # type: ignore
    b = datetime.timezone.utc # type: ignore
    c = datetime.timezone(datetime.timedelta(hours=1)) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = decimal.Decimal(3.14) # type: ignore
    b = decimal.Decimal(3.14) # type: ignore
    c = decimal.Decimal(3.15) # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    class Color(Flag):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    class Color2(Enum):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    a = Color.RED # type: ignore
    b = Color.RED # type: ignore
    c = Color.GREEN # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a = Color2.RED # type: ignore
    b = Color2.RED # type: ignore
    c = Color2.GREEN # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    class Color4(IntFlag):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    a = Color4.RED  # type: ignore
    b = Color4.RED # type: ignore
    c = Color4.GREEN # type: ignore
    assert comparator(a, b)
    assert not comparator(a, c)

    a: re.Pattern = re.compile("a")
    b: re.Pattern = re.compile("a")
    c: re.Pattern = re.compile("b")
    d: re.Pattern = re.compile("a", re.IGNORECASE)
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    arr1 = array.array('i', [1, 2, 3])
    arr2 = array.array('i', [1, 2, 3])
    arr3 = array.array('i', [4, 5, 6])
    arr4 = array.array('f', [1.0, 2.0, 3.0])

    assert comparator(arr1, arr2)
    assert not comparator(arr1, arr3)
    assert not comparator(arr1, arr4)
    assert not comparator(arr1, [1, 2, 3])

    empty_arr_i1 = array.array('i')
    empty_arr_i2 = array.array('i')
    empty_arr_f = array.array('f')
    assert comparator(empty_arr_i1, empty_arr_i2)
    assert not comparator(empty_arr_i1, empty_arr_f)
    assert not comparator(empty_arr_i1, arr1)

    id1 = uuid.uuid4()
    id3 = uuid.uuid4()
    assert comparator(id1, id1)
    assert not comparator(id1, id3)




def test_numpy():
    try:
        import numpy as np
    except ImportError:
        pytest.skip()
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    c = np.array([1, 2, 4])
    assert comparator(a, b)
    assert not comparator(a, c)

    d = np.array([[1, 2], [3, 4]])
    e = np.array([[1, 2], [3, 4]])
    f = np.array([[1, 2], [3, 5]])
    assert comparator(d, e)
    assert not comparator(d, f)
    assert not comparator(a, d)

    g = np.array([1.0, 2.0, 3.0])
    assert not comparator(a, g)

    h = np.float32(1.0)
    i = np.float32(1.0)
    assert comparator(h, i)

    j = np.float64(1.0)
    k = np.float64(1.0)
    assert not comparator(h, j)
    assert comparator(j, k)

    l = np.int32(1)
    m = np.int32(1)
    assert comparator(l, m)
    assert not comparator(l, h)
    assert not comparator(l, j)

    n = np.int64(1)
    o = np.int64(1)
    assert not comparator(n, l)
    assert comparator(n, o)

    p = np.uint32(1)
    q = np.uint32(1)
    assert comparator(p, q)
    assert not comparator(p, l)

    r = np.uint64(1)
    s = np.uint64(1)
    assert not comparator(r, p)
    assert comparator(r, s)

    t = np.bool_(True)
    u = np.bool_(True)
    assert comparator(t, u)
    assert not comparator(t, r)

    v = np.complex64(1.0 + 1.0j)
    w = np.complex64(1.0 + 1.0j)
    assert comparator(v, w)
    assert not comparator(v, t)

    x = np.complex128(1.0 + 1.0j)
    y = np.complex128(1.0 + 1.0j)
    assert not comparator(x, v)
    assert comparator(x, y)

    # Create numpy array with mixed type object
    z = np.array([1, 2, "str"], dtype=np.object_)
    aa = np.array([1, 2, "str"], dtype=np.object_)
    ab = np.array([1, 2, "str2"], dtype=np.object_)
    assert comparator(z, aa)
    assert not comparator(z, ab)

    ac = np.array([1, 2, "str2"])
    ad = np.array([1, 2, "str2"])
    assert comparator(ac, ad)

    # Test for numpy array with nan and inf
    ae = np.array([1, 2, np.nan])
    af = np.array([1, 2, np.nan])
    ag = np.array([1, 2, np.inf])
    ah = np.array([1, 2, np.inf])
    ai = np.inf
    aj = np.inf
    ak = np.nan
    al = np.nan
    assert comparator(ae, af)
    assert comparator(ag, ah)
    assert not comparator(ae, ag)
    assert not comparator(af, ah)
    assert comparator(ai, aj)
    assert comparator(ak, al)
    assert not comparator(ai, ak)

    dt = np.dtype([('name', 'S10'), ('age', np.int32)])
    a_struct = np.array([('Alice', 25)], dtype=dt)
    b_struct = np.array([('Alice', 25)], dtype=dt)
    c_struct = np.array([('Bob', 30)], dtype=dt)

    a_void = a_struct[0]
    b_void = b_struct[0]
    c_void = c_struct[0]

    assert isinstance(a_void, np.void)
    assert comparator(a_void, b_void)
    assert not comparator(a_void, c_void)


def test_numpy_random_generator():
    try:
        import numpy as np
    except ImportError:
        pytest.skip()

    # Test numpy.random.Generator (modern API)
    # Same seed should produce equal generators
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    assert comparator(rng1, rng2)

    # Different seeds should produce non-equal generators
    rng3 = np.random.default_rng(seed=123)
    assert not comparator(rng1, rng3)

    # After generating numbers, state changes
    rng4 = np.random.default_rng(seed=42)
    rng5 = np.random.default_rng(seed=42)
    rng4.random()  # Advance state
    assert not comparator(rng4, rng5)

    # Both advanced by same amount should be equal
    rng5.random()
    assert comparator(rng4, rng5)

    # Test with different bit generators
    from numpy.random import PCG64, MT19937
    rng_pcg1 = np.random.Generator(PCG64(seed=42))
    rng_pcg2 = np.random.Generator(PCG64(seed=42))
    assert comparator(rng_pcg1, rng_pcg2)

    rng_mt1 = np.random.Generator(MT19937(seed=42))
    rng_mt2 = np.random.Generator(MT19937(seed=42))
    assert comparator(rng_mt1, rng_mt2)

    # Different bit generator types should not be equal
    assert not comparator(rng_pcg1, rng_mt1)


def test_numpy_random_state():
    try:
        import numpy as np
    except ImportError:
        pytest.skip()

    # Test numpy.random.RandomState (legacy API)
    # Same seed should produce equal states
    rs1 = np.random.RandomState(seed=42)
    rs2 = np.random.RandomState(seed=42)
    assert comparator(rs1, rs2)

    # Different seeds should produce non-equal states
    rs3 = np.random.RandomState(seed=123)
    assert not comparator(rs1, rs3)

    # After generating numbers, state changes
    rs4 = np.random.RandomState(seed=42)
    rs5 = np.random.RandomState(seed=42)
    rs4.random()  # Advance state
    assert not comparator(rs4, rs5)

    # Both advanced by same amount should be equal
    rs5.random()
    assert comparator(rs4, rs5)

    # Test state restoration
    rs6 = np.random.RandomState(seed=42)
    state = rs6.get_state()
    rs6.random()  # Advance state
    rs7 = np.random.RandomState(seed=42)
    rs7.set_state(state)
    # rs6 advanced, rs7 restored to original state
    assert not comparator(rs6, rs7)


def test_scipy():
    try:
        import scipy as sp  # type: ignore
    except ImportError:
        pytest.skip()
    a = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    b = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    c = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    ca = sp.sparse.csr_matrix([[1, 0, 0, 0], [0, 0, 3, 0], [4, 0, 6, 0]])
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(c, ca)

    d = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    e = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    f = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    fa = sp.sparse.csc_matrix([[1, 0, 0, 0], [0, 0, 3, 0], [4, 0, 6, 0]])
    assert comparator(d, e)
    assert not comparator(d, f)
    assert not comparator(a, d)
    assert not comparator(c, f)
    assert not comparator(f, fa)

    g = sp.sparse.lil_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    h = sp.sparse.lil_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    i = sp.sparse.lil_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(g, h)
    assert not comparator(g, i)
    assert not comparator(a, g)

    j = sp.sparse.dok_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    k = sp.sparse.dok_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    l = sp.sparse.dok_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(j, k)
    assert not comparator(j, l)
    assert not comparator(a, j)

    m = sp.sparse.dia_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    n = sp.sparse.dia_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    o = sp.sparse.dia_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(m, n)
    assert not comparator(m, o)
    assert not comparator(a, m)

    p = sp.sparse.coo_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    q = sp.sparse.coo_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    r = sp.sparse.coo_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(p, q)
    assert not comparator(p, r)
    assert not comparator(a, p)

    s = sp.sparse.bsr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    t = sp.sparse.bsr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    u = sp.sparse.bsr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(s, t)
    assert not comparator(s, u)
    assert not comparator(a, s)

    try:
        import numpy as np

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        v = sp.sparse.coo_array((data, (row, col)), shape=(4, 4)).toarray()
        w = sp.sparse.coo_array((data, (row, col)), shape=(4, 4)).toarray()
        assert comparator(v, w)
    except ImportError:
        print("Should run tests with numpy installed to test more thoroughly")


def test_pandas():
    try:
        import pandas as pd
    except ImportError:
        pytest.skip()
    a = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    b = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    c = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    ca = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(c, ca)

    ak = pd.DataFrame(
        {"a": [datetime.datetime(2020, 2, 2, 2, 2, 2), datetime.datetime(2020, 2, 2, 2, 2, 2)], "b": [4, 5]}
    )
    al = pd.DataFrame(
        {"a": [datetime.datetime(2020, 2, 2, 2, 2, 2), datetime.datetime(2020, 2, 2, 2, 2, 2)], "b": [4, 5]}
    )
    am = pd.DataFrame(
        {"a": [datetime.datetime(2020, 2, 2, 2, 2, 2), datetime.datetime(2020, 2, 2, 2, 2, 3)], "b": [4, 5]}
    )
    assert comparator(ak, al)
    assert not comparator(ak, am)

    d = pd.Series([1, 2, 3])
    e = pd.Series([1, 2, 3])
    f = pd.Series([1, 2, 4])
    assert comparator(d, e)
    assert not comparator(d, f)

    g = pd.Index([1, 2, 3])
    h = pd.Index([1, 2, 3])
    i = pd.Index([1, 2, 4])
    assert comparator(g, h)
    assert not comparator(g, i)

    j = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    k = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    l = pd.MultiIndex.from_tuples([(1, 2), (3, 5)])
    assert comparator(j, k)
    assert not comparator(j, l)

    m = pd.Categorical([1, 2, 3])
    n = pd.Categorical([1, 2, 3])
    o = pd.Categorical([1, 2, 4])
    assert comparator(m, n)
    assert not comparator(m, o)

    p = pd.Interval(1, 2)
    q = pd.Interval(1, 2)
    r = pd.Interval(1, 3)
    assert comparator(p, q)
    assert not comparator(p, r)

    s = pd.IntervalIndex.from_tuples([(1, 2), (3, 4)])
    t = pd.IntervalIndex.from_tuples([(1, 2), (3, 4)])
    u = pd.IntervalIndex.from_tuples([(1, 2), (3, 5)])
    assert comparator(s, t)
    assert not comparator(s, u)

    v = pd.Period("2021-01")
    w = pd.Period("2021-01")
    x = pd.Period("2021-02")
    assert comparator(v, w)
    assert not comparator(v, x)

    y = pd.period_range(start="2021-01", periods=3, freq="M")
    z = pd.period_range(start="2021-01", periods=3, freq="M")
    aa = pd.period_range(start="2021-01", periods=4, freq="M")
    assert comparator(y, z)
    assert not comparator(y, aa)

    ab = pd.Timedelta("1 days")
    ac = pd.Timedelta("1 days")
    ad = pd.Timedelta("2 days")
    assert comparator(ab, ac)
    assert not comparator(ab, ad)

    ae = pd.TimedeltaIndex(["1 days", "2 days"])
    af = pd.TimedeltaIndex(["1 days", "2 days"])
    ag = pd.TimedeltaIndex(["1 days", "3 days"])
    assert comparator(ae, af)
    assert not comparator(ae, ag)

    ah = pd.Timestamp("2021-01-01")
    ai = pd.Timestamp("2021-01-01")
    aj = pd.Timestamp("2021-01-02")
    assert comparator(ah, ai)
    assert not comparator(ah, aj)

    # test cases for sparse pandas arrays
    an = pd.arrays.SparseArray([1, 2, 3])
    ao = pd.arrays.SparseArray([1, 2, 3])
    ap = pd.arrays.SparseArray([1, 2, 4])
    assert comparator(an, ao)
    assert not comparator(an, ap)

    assert comparator(pd.NA, pd.NA)
    assert not comparator(pd.NA, None)
    assert not comparator(None, pd.NA)

    s1 = pd.Series([1, 2, pd.NA, 4])
    s2 = pd.Series([1, 2, pd.NA, 4])
    s3 = pd.Series([1, 2, None, 4])

    assert comparator(s1, s2)
    assert not comparator(s1, s3)

    df1 = pd.DataFrame({'a': [1, 2, pd.NA], 'b': [4, pd.NA, 6]})
    df2 = pd.DataFrame({'a': [1, 2, pd.NA], 'b': [4, pd.NA, 6]})
    df3 = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
    assert comparator(df1, df2)
    assert not comparator(df1, df3)

    d1 = {'a': pd.NA, 'b': [1, pd.NA, 3]}
    d2 = {'a': pd.NA, 'b': [1, pd.NA, 3]}
    d3 = {'a': None, 'b': [1, None, 3]}
    assert comparator(d1, d2)
    assert not comparator(d1, d3)

    s1 = pd.Series([1, 2, pd.NA, 4])
    s2 = pd.Series([1, 2, pd.NA, 4])

    filtered1 = s1[s1 > 1]
    filtered2 = s2[s2 > 1]
    assert comparator(filtered1, filtered2)


def test_pyrsistent():
    try:
        from pyrsistent import PBag, PClass, PRecord, field, pdeque, pmap, pset, pvector  # type: ignore
    except ImportError:
        pytest.skip()

    a = pmap({"a": 1, "b": 2})
    b = pmap({"a": 1, "b": 2})
    c = pmap({"a": 1, "b": 3})
    assert comparator(a, b)
    assert not comparator(a, c)

    d = pvector([1, 2, 3])
    e = pvector([1, 2, 3])
    f = pvector([1, 2, 4])
    assert comparator(d, e)
    assert not comparator(d, f)

    g = pset([1, 2, 3])
    h = pset([2, 3, 1])
    i = pset([1, 2, 4])
    assert comparator(g, h)
    assert not comparator(g, i)

    class TestRecord(PRecord):
        a = field()
        b = field()

    j = TestRecord()
    k = TestRecord()
    l = TestRecord(a=2, b=3)
    assert comparator(j, k)
    assert not comparator(j, l)

    class TestClass(PClass):
        a = field()
        b = field()

    m = TestClass()
    n = TestClass()
    o = TestClass(a=1, b=3)
    assert comparator(m, n)
    assert not comparator(m, o)

    p = pdeque([1, 2, 3], 3)
    q = pdeque([1, 2, 3], 3)
    r = pdeque([1, 2, 4], 3)
    assert comparator(p, q)
    assert not comparator(p, r)

    s = PBag([1, 2, 3])
    t = PBag([1, 2, 3])
    u = PBag([1, 2, 4])
    assert comparator(s, t)
    assert not comparator(s, u)

    v = pvector([1, 2, 3])
    w = pvector([1, 2, 3])
    x = pvector([1, 2, 4])
    assert comparator(v, w)
    assert not comparator(v, x)


def test_torch_dtype():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip()

    # Test torch.dtype comparisons
    a = torch.float32
    b = torch.float32
    c = torch.float64
    d = torch.int32
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # Test different dtype categories
    e = torch.int64
    f = torch.int64
    g = torch.int32
    assert comparator(e, f)
    assert not comparator(e, g)

    # Test complex dtypes
    h = torch.complex64
    i = torch.complex64
    j = torch.complex128
    assert comparator(h, i)
    assert not comparator(h, j)

    # Test bool dtype
    k = torch.bool
    l = torch.bool
    m = torch.int8
    assert comparator(k, l)
    assert not comparator(k, m)


def test_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip()

    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3])
    c = torch.tensor([1, 2, 4])
    assert comparator(a, b)
    assert not comparator(a, c)

    d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    e = torch.tensor([[1, 2, 3], [4, 5, 6]])
    f = torch.tensor([[1, 2, 3], [4, 5, 7]])
    assert comparator(d, e)
    assert not comparator(d, f)

    # Test tensors with different data types
    g = torch.tensor([1, 2, 3], dtype=torch.float32)
    h = torch.tensor([1, 2, 3], dtype=torch.float32)
    i = torch.tensor([1, 2, 3], dtype=torch.int64)
    assert comparator(g, h)
    assert not comparator(g, i)

    # Test 3D tensors
    j = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    l = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 9]]])
    assert comparator(j, k)
    assert not comparator(j, l)

    # Test tensors with different shapes
    m = torch.tensor([1, 2, 3])
    n = torch.tensor([[1, 2, 3]])
    assert not comparator(m, n)

    # Test empty tensors
    o = torch.tensor([])
    p = torch.tensor([])
    q = torch.tensor([1])
    assert comparator(o, p)
    assert not comparator(o, q)

    # Test tensors with NaN values
    r = torch.tensor([1.0, float('nan'), 3.0])
    s = torch.tensor([1.0, float('nan'), 3.0])
    t = torch.tensor([1.0, 2.0, 3.0])
    assert comparator(r, s)  # NaN == NaN
    assert not comparator(r, t)

    # Test tensors with infinity values
    u = torch.tensor([1.0, float('inf'), 3.0])
    v = torch.tensor([1.0, float('inf'), 3.0])
    w = torch.tensor([1.0, float('-inf'), 3.0])
    assert comparator(u, v)
    assert not comparator(u, w)

    # Test tensors with different devices (if CUDA is available)
    if torch.cuda.is_available():
        x = torch.tensor([1, 2, 3]).cuda()
        y = torch.tensor([1, 2, 3]).cuda()
        z = torch.tensor([1, 2, 3])
        assert comparator(x, y)
        assert not comparator(x, z)

    # Test tensors with requires_grad
    aa = torch.tensor([1., 2., 3.], requires_grad=True)
    bb = torch.tensor([1., 2., 3.], requires_grad=True)
    cc = torch.tensor([1., 2., 3.], requires_grad=False)
    assert comparator(aa, bb)
    assert not comparator(aa, cc)

    # Test complex tensors
    dd = torch.tensor([1+2j, 3+4j])
    ee = torch.tensor([1+2j, 3+4j])
    ff = torch.tensor([1+2j, 3+5j])
    assert comparator(dd, ee)
    assert not comparator(dd, ff)

    # Test boolean tensors
    gg = torch.tensor([True, False, True])
    hh = torch.tensor([True, False, True])
    ii = torch.tensor([True, True, True])
    assert comparator(gg, hh)
    assert not comparator(gg, ii)


def test_torch_device():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip()

    # Test torch.device comparisons - same device type
    a = torch.device("cpu")
    b = torch.device("cpu")
    assert comparator(a, b)

    # Test different device types
    c = torch.device("cpu")
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        assert not comparator(c, d)

    # Test device with index
    e = torch.device("cpu")
    f = torch.device("cpu")
    assert comparator(e, f)

    # Test cuda devices with different indices (if multiple GPUs available)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        g = torch.device("cuda:0")
        h = torch.device("cuda:0")
        i = torch.device("cuda:1")
        assert comparator(g, h)
        assert not comparator(g, i)

    # Test cuda device with and without explicit index
    if torch.cuda.is_available():
        j = torch.device("cuda:0")
        k = torch.device("cuda", 0)
        assert comparator(j, k)

    # Test meta device
    l = torch.device("meta")
    m = torch.device("meta")
    n = torch.device("cpu")
    assert comparator(l, m)
    assert not comparator(l, n)


def test_jax():
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip()

    # Test basic arrays
    a = jnp.array([1, 2, 3])
    b = jnp.array([1, 2, 3])
    c = jnp.array([1, 2, 4])
    assert comparator(a, b)
    assert not comparator(a, c)

    # Test 2D arrays
    d = jnp.array([[1, 2, 3], [4, 5, 6]])
    e = jnp.array([[1, 2, 3], [4, 5, 6]])
    f = jnp.array([[1, 2, 3], [4, 5, 7]])
    assert comparator(d, e)
    assert not comparator(d, f)

    # Test arrays with different data types
    g = jnp.array([1, 2, 3], dtype=jnp.float32)
    h = jnp.array([1, 2, 3], dtype=jnp.float32)
    i = jnp.array([1, 2, 3], dtype=jnp.int32)
    assert comparator(g, h)
    assert not comparator(g, i)

    # Test 3D arrays
    j = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    l = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 9]]])
    assert comparator(j, k)
    assert not comparator(j, l)

    # Test arrays with different shapes
    m = jnp.array([1, 2, 3])
    n = jnp.array([[1, 2, 3]])
    assert not comparator(m, n)

    # Test empty arrays
    o = jnp.array([])
    p = jnp.array([])
    q = jnp.array([1])
    assert comparator(o, p)
    assert not comparator(o, q)

    # Test arrays with NaN values
    r = jnp.array([1.0, jnp.nan, 3.0])
    s = jnp.array([1.0, jnp.nan, 3.0])
    t = jnp.array([1.0, 2.0, 3.0])
    assert comparator(r, s)  # NaN == NaN
    assert not comparator(r, t)

    # Test arrays with infinity values
    u = jnp.array([1.0, jnp.inf, 3.0])
    v = jnp.array([1.0, jnp.inf, 3.0])
    w = jnp.array([1.0, -jnp.inf, 3.0])
    assert comparator(u, v)
    assert not comparator(u, w)

    # Test complex arrays
    x = jnp.array([1+2j, 3+4j])
    y = jnp.array([1+2j, 3+4j])
    z = jnp.array([1+2j, 3+5j])
    assert comparator(x, y)
    assert not comparator(x, z)

    # Test boolean arrays
    aa = jnp.array([True, False, True])
    bb = jnp.array([True, False, True])
    cc = jnp.array([True, True, True])
    assert comparator(aa, bb)
    assert not comparator(aa, cc)


def test_xarray():
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        pytest.skip()

    # Test basic DataArray
    a = xr.DataArray([1, 2, 3], dims=['x'])
    b = xr.DataArray([1, 2, 3], dims=['x'])
    c = xr.DataArray([1, 2, 4], dims=['x'])
    assert comparator(a, b)
    assert not comparator(a, c)

    # Test DataArray with coordinates
    d = xr.DataArray([1, 2, 3], coords={'x': [0, 1, 2]}, dims=['x'])
    e = xr.DataArray([1, 2, 3], coords={'x': [0, 1, 2]}, dims=['x'])
    f = xr.DataArray([1, 2, 3], coords={'x': [0, 1, 3]}, dims=['x'])
    assert comparator(d, e)
    assert not comparator(d, f)

    # Test DataArray with attributes
    g = xr.DataArray([1, 2, 3], dims=['x'], attrs={'units': 'meters'})
    h = xr.DataArray([1, 2, 3], dims=['x'], attrs={'units': 'meters'})
    i = xr.DataArray([1, 2, 3], dims=['x'], attrs={'units': 'feet'})
    assert comparator(g, h)
    assert not comparator(g, i)

    # Test 2D DataArray
    j = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=['x', 'y'])
    k = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=['x', 'y'])
    l = xr.DataArray([[1, 2, 3], [4, 5, 7]], dims=['x', 'y'])
    assert comparator(j, k)
    assert not comparator(j, l)

    # Test DataArray with different dimensions
    m = xr.DataArray([1, 2, 3], dims=['x'])
    n = xr.DataArray([1, 2, 3], dims=['y'])
    assert not comparator(m, n)

    # Test DataArray with NaN values
    o = xr.DataArray([1.0, np.nan, 3.0], dims=['x'])
    p = xr.DataArray([1.0, np.nan, 3.0], dims=['x'])
    q = xr.DataArray([1.0, 2.0, 3.0], dims=['x'])
    assert comparator(o, p)
    assert not comparator(o, q)

    # Test Dataset
    r = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]]),
        'pressure': (['x', 'y'], [[5, 6], [7, 8]])
    })
    s = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]]),
        'pressure': (['x', 'y'], [[5, 6], [7, 8]])
    })
    t = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]]),
        'pressure': (['x', 'y'], [[5, 6], [7, 9]])
    })
    assert comparator(r, s)
    assert not comparator(r, t)

    # Test Dataset with coordinates
    u = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]])
    }, coords={'x': [0, 1], 'y': [0, 1]})
    v = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]])
    }, coords={'x': [0, 1], 'y': [0, 1]})
    w = xr.Dataset({
        'temp': (['x', 'y'], [[1, 2], [3, 4]])
    }, coords={'x': [0, 2], 'y': [0, 1]})
    assert comparator(u, v)
    assert not comparator(u, w)

    # Test Dataset with attributes
    x = xr.Dataset({'temp': (['x'], [1, 2, 3])}, attrs={'source': 'sensor'})
    y = xr.Dataset({'temp': (['x'], [1, 2, 3])}, attrs={'source': 'sensor'})
    z = xr.Dataset({'temp': (['x'], [1, 2, 3])}, attrs={'source': 'model'})
    assert comparator(x, y)
    assert not comparator(x, z)

    # Test Dataset with different variables
    aa = xr.Dataset({'temp': (['x'], [1, 2, 3])})
    bb = xr.Dataset({'temp': (['x'], [1, 2, 3])})
    cc = xr.Dataset({'pressure': (['x'], [1, 2, 3])})
    assert comparator(aa, bb)
    assert not comparator(aa, cc)

    # Test empty Dataset
    dd = xr.Dataset()
    ee = xr.Dataset()
    assert comparator(dd, ee)

    # Test DataArray with different shapes
    ff = xr.DataArray([1, 2, 3], dims=['x'])
    gg = xr.DataArray([[1, 2, 3]], dims=['x', 'y'])
    assert not comparator(ff, gg)

    # Test DataArray with different data types
    # Note: xarray.identical() considers int and float arrays with same values as identical
    hh = xr.DataArray(np.array([1, 2, 3], dtype='int32'), dims=['x'])
    ii = xr.DataArray(np.array([1, 2, 3], dtype='int64'), dims=['x'])
    # xarray is permissive with dtype comparisons, treats these as identical
    assert comparator(hh, ii)

    # Test DataArray with infinity
    jj = xr.DataArray([1.0, np.inf, 3.0], dims=['x'])
    kk = xr.DataArray([1.0, np.inf, 3.0], dims=['x'])
    ll = xr.DataArray([1.0, -np.inf, 3.0], dims=['x'])
    assert comparator(jj, kk)
    assert not comparator(jj, ll)

    # Test Dataset vs DataArray (different types)
    mm = xr.DataArray([1, 2, 3], dims=['x'])
    nn = xr.Dataset({'data': (['x'], [1, 2, 3])})
    assert not comparator(mm, nn)


def test_returns():
    a = Success(5)
    b = Success(5)
    c = Success(6)
    d = Failure(5)
    e = Success((5, 5))
    f = Success((5, 6))
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(a, e)
    assert not comparator(e, f)

    g = Success((5, 5))
    h = Success((5, 5))
    i = Success((5, 6))
    assert comparator(g, h)
    assert not comparator(g, i)


def test_custom_object():
    class TestClass:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return self.value == other.value

    a = TestClass(5)
    b = TestClass(5)
    c = TestClass(6)
    assert comparator(a, b)
    assert not comparator(a, c)

    class TestClass2:
        def __init__(self, value1, value2=6):
            self.value1 = value1
            self.value2 = value2

    a = TestClass(5)
    b = TestClass2(5, 6)
    c = TestClass2(5, 7)
    d = TestClass2(5, 6)
    assert not comparator(a, b)
    assert not comparator(b, c)
    assert comparator(b, d)

    class TestClass3(TestClass):
        def print(self):
            print(self.value)

    a = TestClass2(5)
    b = TestClass3(5)
    c = TestClass3(5)
    assert not comparator(a, b)
    assert comparator(b, c)

    @dataclasses.dataclass
    class InventoryItem:
        """Class for keeping track of an item in inventory."""

        name: str
        unit_price: float
        quantity_on_hand: int = 0

        def total_cost(self) -> float:
            return self.unit_price * self.quantity_on_hand

    a = InventoryItem(name="widget", unit_price=3.0, quantity_on_hand=10)
    b = InventoryItem(name="widget", unit_price=3.0, quantity_on_hand=10)
    c = InventoryItem(name="widget", unit_price=3.0, quantity_on_hand=11)

    assert comparator(a, b)
    assert not comparator(a, c)

    @pydantic.dataclasses.dataclass
    class InventoryItemPydantic:
        """Class for keeping track of an item in inventory."""

        name: str
        unit_price: float
        quantity_on_hand: int = 0

        def total_cost(self) -> float:
            return self.unit_price * self.quantity_on_hand

    a = InventoryItemPydantic(name="widget", unit_price=3.0, quantity_on_hand=10)
    b = InventoryItemPydantic(name="widget", unit_price=3.0, quantity_on_hand=10)
    c = InventoryItemPydantic(name="widget", unit_price=3.0, quantity_on_hand=11)
    assert comparator(a, b)
    assert not comparator(a, c)

    class InventoryItemBasePydantic(pydantic.BaseModel):
        name: str
        unit_price: float
        quantity_on_hand: int = 0

        def total_cost(self) -> float:
            return self.unit_price * self.quantity_on_hand

    a = InventoryItemBasePydantic(name="widget", unit_price=3.0, quantity_on_hand=10)
    b = InventoryItemBasePydantic(name="widget", unit_price=3.0, quantity_on_hand=10)
    c = InventoryItemBasePydantic(name="widget", unit_price=3.0, quantity_on_hand=11)
    assert comparator(a, b)
    assert not comparator(a, c)

    class A:
        items = [1, 2, 3]
        val = 5

    class B:
        items = [1, 2, 4]
        val = 5

    assert comparator(A, A)
    assert not comparator(A, B)

    class C:
        items = [1, 2, 3]
        val = 5

        def __init__(self):
            self.itemm2 = [1, 2, 3]
            self.val2 = 5

    class D:
        items = [1, 2, 3]
        val = 5

        def __init__(self):
            self.itemm2 = [1, 2, 4]
            self.val2 = 5

    assert comparator(C, C)
    assert not comparator(C, D)

    E = C
    assert comparator(C, E)


def test_custom_object_2():
    fto_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_method.py").resolve()
    original_code = fto_path.read_text("utf-8")
    from code_to_optimize.bubble_sort_method import BubbleSorter

    a = BubbleSorter()
    assert a.x == 0
    try:
        # Remove the module from sys.modules, to get the updated class
        sys.modules.pop("code_to_optimize.bubble_sort_method", None)
        from code_to_optimize.bubble_sort_method import BubbleSorter

        b = BubbleSorter()
        assert comparator(
            a, b
        )  # Note that type(a) != type(b) as the class type objects are different, even if the code is the same.

        optimized_code_mutated_attr = """
class BubbleSorter:
    z = 0

    def __init__(self, x=1):
        self.x = x

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                                    """
        fto_path.write_text(optimized_code_mutated_attr, "utf-8")
        sys.modules.pop("code_to_optimize.bubble_sort_method", None)
        from code_to_optimize.bubble_sort_method import BubbleSorter

        c = BubbleSorter()
        assert c.x == 1
        assert not comparator(a, c)

        optimized_code_new_attr = """
class BubbleSorter:
    z = 5

    def __init__(self, x=0):
        self.x = x

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
                                            """
        fto_path.write_text(optimized_code_new_attr, "utf-8")
        sys.modules.pop("code_to_optimize.bubble_sort_method", None)
        from code_to_optimize.bubble_sort_method import BubbleSorter

        d = BubbleSorter()
        assert d.x == 0
        # Currently, we do not check if class variables are different, since the code replacer does not allow this.
        # In the future, if this functionality is allowed, this assert should be false.
        assert comparator(a, d)
    finally:
        fto_path.write_text(original_code, "utf-8")


def test_superset():
    class A:
        def __init__(self):
            self.a = 1

    obj = A()
    obj.x = 3

    assert comparator(A(), obj, superset_obj=True)
    assert not comparator(obj, A(), superset_obj=True)
    assert not comparator(A(), obj)
    assert not comparator(obj, A())
    assert comparator(obj, obj, superset_obj=True)
    assert comparator(obj, obj)


def test_compare_results_fn():
    original_results = TestResults()
    original_results.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    new_results_1 = TestResults()
    new_results_1.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=10,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(original_results, new_results_1)
    assert match

    new_results_2 = TestResults()
    new_results_2.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=10,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=[5],
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(original_results, new_results_2)
    assert not match

    new_results_3 = TestResults()
    new_results_3.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=10,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )
    new_results_3.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="2",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=10,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(original_results, new_results_3)
    assert match

    new_results_4 = TestResults()
    new_results_4.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=False,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(original_results, new_results_4)
    assert not match

    new_results_5_baseline = TestResults()
    new_results_5_baseline.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    new_results_5_opt = TestResults()
    new_results_5_opt.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=False,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.GENERATED_REGRESSION,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(new_results_5_baseline, new_results_5_opt)
    assert not match

    new_results_6_baseline = TestResults()
    new_results_6_baseline.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=True,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.REPLAY_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    new_results_6_opt = TestResults()
    new_results_6_opt.add(
        FunctionTestInvocation(
            id=InvocationId(
                test_module_path="test_module_path",
                test_class_name="test_class_name",
                test_function_name="test_function_name",
                function_getting_tested="function_getting_tested",
                iteration_id="0",
            ),
            file_name=Path("file_name"),
            did_pass=False,
            runtime=5,
            test_framework="unittest",
            test_type=TestType.REPLAY_TEST,
            return_value=5,
            timed_out=False,
            loop_index=1,
        )
    )

    match, _ = compare_test_results(new_results_6_baseline, new_results_6_opt)
    assert not match

    match, _ = compare_test_results(TestResults(), TestResults())
    assert not match


def test_exceptions():
    type_error = TypeError("This is a type error")

    type_error_2 = TypeError("This is a type error")

    assert comparator(type_error, type_error_2)


def raise_exception():
    raise Exception("This is an exception")


def test_exceptions_comparator():
    # Currently we are only comparing the exception types and the attributes that don't start with "_"
    # there are complications with comparing the exception messages
    try:
        raise_exception()
    except Exception as e:
        exception = e

    try:
        raise_exception()
    except Exception as b:
        exception_2 = b

    assert comparator(exception, exception_2)

    exc1 = ValueError("same message")
    exc2 = ValueError("same message")
    assert comparator(exc1, exc2)

    exc_msg1 = ValueError("message one")
    exc_msg2 = ValueError("message two")
    # Different messages but same types
    assert comparator(exc_msg1, exc_msg2)

    exc1 = ValueError("common message")
    exc2 = TypeError("common message")
    assert not comparator(exc1, exc2)

    exc_file_1 = FileNotFoundError(2, "No such file or directory")

    exc_file2 = FileNotFoundError(2, "No such file or directory")

    exc_file4 = FileNotFoundError(2, "File not found")

    exc_file3 = FileNotFoundError(3, "No such file or directory")

    assert not comparator(exc1, exc2)

    assert comparator(exc_file_1, exc_file2)

    assert comparator(exc_file_1, exc_file3)

    assert comparator(exc_file_1, exc_file4)

    assert comparator(exception, exception)

    assert not comparator(exception, None)
    assert not comparator(None, exception)
    assert comparator(None, None)

    # Different exception types
    exc_type1 = TypeError("Type error")
    exc_type2 = TypeError("Another type error")
    assert comparator(exc_type1, exc_type2)

    exc_type3 = KeyError("Missing key")
    exc_type4 = KeyError("Missing key")
    assert comparator(exc_type3, exc_type4)
    assert not comparator(exc_type1, exc_type3)

    # compare the attributes of the exception as well
    class CustomError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    custom_exc1 = CustomError("Something went wrong", 101)
    custom_exc2 = CustomError("Something went wrong", 101)
    assert comparator(custom_exc1, custom_exc2)

    custom_exc4 = CustomError("Something went wrong", 102)

    assert not comparator(custom_exc1, custom_exc4)

    class CustomErrorNoArgs(Exception):
        pass

    custom_no_args1 = CustomErrorNoArgs()
    custom_no_args2 = CustomErrorNoArgs()
    assert comparator(custom_no_args1, custom_no_args2)

    exc_empty1 = ValueError("")
    exc_empty2 = ValueError("")
    assert comparator(exc_empty1, exc_empty2)

    exc_not_empty = ValueError("Not empty")
    assert comparator(exc_empty1, exc_not_empty)

    class CustomValueError(ValueError):
        pass

    custom_value_error1 = CustomValueError("A custom value error")
    value_error1 = ValueError("A custom value error")
    assert not comparator(custom_value_error1, value_error1)

    custom_value_error2 = CustomValueError("Another custom value error")
    assert comparator(custom_value_error1, custom_value_error2)

    class CustomExceptionWithArgs(Exception):
        def __init__(self, arg1, arg2):
            self.args = (arg1, arg2)

    custom_args_exc1 = CustomExceptionWithArgs(1, "test")
    custom_args_exc2 = CustomExceptionWithArgs(1, "test")
    assert comparator(custom_args_exc1, custom_args_exc2)

    custom_args_exc3 = CustomExceptionWithArgs(1, "different")
    assert comparator(custom_args_exc1, custom_args_exc3)

    def raise_specific_exception():
        raise ZeroDivisionError("Cannot divide by zero")

    try:
        raise_specific_exception()
    except ZeroDivisionError as z1:
        zero_division_exc1 = z1

    try:
        raise_specific_exception()
    except ZeroDivisionError as z2:
        zero_division_exc2 = z2

    assert comparator(zero_division_exc1, zero_division_exc2)

    zero_division_exc3 = ZeroDivisionError("Different message")
    assert comparator(zero_division_exc1, zero_division_exc3)

    assert comparator(..., ...)
    assert comparator(Ellipsis, Ellipsis)

    assert not comparator(..., None)

    assert not comparator(Ellipsis, None)

    code7 = "a = 1 + 2"
    module7 = ast.parse(code7)
    for node in ast.walk(module7):
       for child in ast.iter_child_nodes(node):
         child.parent = node # type: ignore
    module8 = copy.deepcopy(module7)
    assert comparator(module7, module8)

    code2 = "a = 1 + 3"

    module2 = ast.parse(code2)

    assert not comparator(module7, module2)


def test_torch_runtime_error_wrapping():
    """Test that TorchRuntimeError wrapping is handled correctly.

    When torch.compile is used, exceptions are wrapped in TorchRuntimeError.
    The comparator should consider an IndexError equivalent to a TorchRuntimeError
    that wraps an IndexError.
    """
    # Create a mock TorchRuntimeError class that mimics torch._dynamo.exc.TorchRuntimeError
    class TorchRuntimeError(Exception):
        """Mock TorchRuntimeError for testing."""

        pass

    # Monkey-patch the __module__ to match torch._dynamo.exc
    TorchRuntimeError.__module__ = "torch._dynamo.exc"

    # Test 1: TorchRuntimeError with __cause__ set to the same exception type
    index_error = IndexError("index 0 is out of bounds for dimension 0 with size 0")
    torch_error = TorchRuntimeError(
        "Dynamo failed to run FX node with fake tensors: got IndexError('index 0 is out of bounds')"
    )
    torch_error.__cause__ = IndexError("index 0 is out of bounds for dimension 0 with size 0")

    # These should be considered equivalent since TorchRuntimeError wraps IndexError
    assert comparator(index_error, torch_error)
    assert comparator(torch_error, index_error)

    # Test 2: TorchRuntimeError without __cause__ but with matching error type in message
    torch_error_no_cause = TorchRuntimeError(
        "Dynamo failed to run FX node with fake tensors: got IndexError('index 0 is out of bounds')"
    )
    assert comparator(index_error, torch_error_no_cause)
    assert comparator(torch_error_no_cause, index_error)

    # Test 3: Different exception types should not be equivalent
    value_error = ValueError("some value error")
    torch_error_index = TorchRuntimeError("got IndexError('some error')")
    torch_error_index.__cause__ = IndexError("some error")
    assert not comparator(value_error, torch_error_index)
    assert not comparator(torch_error_index, value_error)

    # Test 4: TorchRuntimeError wrapping a different type should not match
    type_error = TypeError("some type error")
    torch_error_with_index = TorchRuntimeError("got IndexError('index error')")
    torch_error_with_index.__cause__ = IndexError("index error")
    assert not comparator(type_error, torch_error_with_index)

    # Test 5: Two TorchRuntimeErrors wrapping the same exception type
    torch_error1 = TorchRuntimeError("got IndexError('error 1')")
    torch_error1.__cause__ = IndexError("error 1")
    torch_error2 = TorchRuntimeError("got IndexError('error 2')")
    torch_error2.__cause__ = IndexError("error 2")
    assert comparator(torch_error1, torch_error2)

    # Test 6: Regular exception comparison still works
    error1 = IndexError("same error")
    error2 = IndexError("same error")
    assert comparator(error1, error2)

    # Test 7: Exception wrapped in tuple (return value scenario from debug output)
    orig_return = (
        ("tensor1", "tensor2"),
        {},
        IndexError("index 0 is out of bounds for dimension 0 with size 0"),
    )
    torch_wrapped_return = (
        ("tensor1", "tensor2"),
        {},
        TorchRuntimeError("Dynamo failed: got IndexError('index 0 is out of bounds for dimension 0 with size 0')"),
    )
    torch_wrapped_return[2].__cause__ = IndexError("index 0 is out of bounds for dimension 0 with size 0")
    assert comparator(orig_return, torch_wrapped_return)


def test_extract_exception_from_message():
    """Test the _extract_exception_from_message helper function."""
    # Test with single-quoted message
    result = _extract_exception_from_message("got IndexError('some error message')")
    assert result is not None
    assert isinstance(result, IndexError)

    # Test with double-quoted message
    result = _extract_exception_from_message('got ValueError("another error")')
    assert result is not None
    assert isinstance(result, ValueError)

    # Test with various builtin exception types
    for exc_name, exc_class in [
        ("TypeError", TypeError),
        ("KeyError", KeyError),
        ("RuntimeError", RuntimeError),
        ("AttributeError", AttributeError),
        ("ZeroDivisionError", ZeroDivisionError),
    ]:
        result = _extract_exception_from_message(f"got {exc_name}('test')")
        assert result is not None
        assert isinstance(result, exc_class)

    # Test with no matching pattern
    result = _extract_exception_from_message("This is a normal error message")
    assert result is None

    # Test with non-exception class name
    result = _extract_exception_from_message("got SomeRandomClass('not an exception')")
    assert result is None

    # Test with partial match (no opening quote)
    result = _extract_exception_from_message("got IndexError without quotes")
    assert result is None

    # Test with empty string
    result = _extract_exception_from_message("")
    assert result is None

    # Test with torch-like error message format
    result = _extract_exception_from_message(
        "Dynamo failed to run FX node with fake tensors: got IndexError('index 0 is out of bounds for dimension 0 with size 0')"
    )
    assert result is not None
    assert isinstance(result, IndexError)


def test_get_wrapped_exception():
    """Test the _get_wrapped_exception helper function."""
    # Test with __cause__ (explicit chaining)
    inner_error = ValueError("inner error")
    outer_error = RuntimeError("outer error")
    outer_error.__cause__ = inner_error
    result = _get_wrapped_exception(outer_error)
    assert result is inner_error

    # Test with no wrapping
    plain_error = ValueError("plain error")
    result = _get_wrapped_exception(plain_error)
    assert result is None

    # Test with message pattern
    error_with_pattern = RuntimeError("got TypeError('some type error')")
    result = _get_wrapped_exception(error_with_pattern)
    assert result is not None
    assert isinstance(result, TypeError)

    # Test that __cause__ takes precedence over message pattern
    actual_cause = IndexError("actual cause")
    error_with_both = RuntimeError("got TypeError('different error in message')")
    error_with_both.__cause__ = actual_cause
    result = _get_wrapped_exception(error_with_both)
    assert result is actual_cause
    assert isinstance(result, IndexError)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup requires Python 3.11+")
def test_get_wrapped_exception_exception_group():
    """Test _get_wrapped_exception with ExceptionGroup (Python 3.11+)."""
    # ExceptionGroup with single exception
    inner_error = ValueError("single inner error")
    group = ExceptionGroup("group", [inner_error])
    result = _get_wrapped_exception(group)
    assert result is inner_error

    # ExceptionGroup with multiple exceptions - should return None
    error1 = ValueError("error 1")
    error2 = TypeError("error 2")
    multi_group = ExceptionGroup("multi group", [error1, error2])
    result = _get_wrapped_exception(multi_group)
    assert result is None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup requires Python 3.11+")
def test_comparator_with_exception_group():
    """Test comparator with ExceptionGroup wrapping (Python 3.11+)."""
    # ExceptionGroup wrapping a single ValueError should match a plain ValueError
    inner_value_error = ValueError("some value error")
    group = ExceptionGroup("group", [inner_value_error])

    plain_value_error = ValueError("different message but same type")
    assert comparator(group, plain_value_error)
    assert comparator(plain_value_error, group)

    # ExceptionGroup with different exception type should not match
    inner_type_error = TypeError("type error")
    type_group = ExceptionGroup("group", [inner_type_error])
    assert not comparator(type_group, plain_value_error)

    # Two ExceptionGroups with same wrapped type should match
    group1 = ExceptionGroup("group1", [ValueError("error 1")])
    group2 = ExceptionGroup("group2", [ValueError("error 2")])
    assert comparator(group1, group2)


def test_comparator_with_cause_chaining():
    """Test comparator with __cause__ exception chaining."""
    # Create an exception chain using 'raise from'
    inner = IndexError("inner index error")
    outer = RuntimeError("outer runtime error")
    outer.__cause__ = inner

    # Outer exception should match the inner exception type
    plain_index_error = IndexError("different index error")
    assert comparator(outer, plain_index_error)
    assert comparator(plain_index_error, outer)

    # Should not match a different type
    plain_type_error = TypeError("type error")
    assert not comparator(outer, plain_type_error)

    # Two chained exceptions with same wrapper type match (regardless of inner type)
    # because same-type exceptions compare non-private attributes only (__cause__ is ignored)
    outer1 = RuntimeError("outer 1")
    outer1.__cause__ = ValueError("inner 1")
    outer2 = RuntimeError("outer 2")
    outer2.__cause__ = ValueError("inner 2")
    assert comparator(outer1, outer2)

    # Different wrapper types with same inner type - unwrapping makes them match
    class WrapperA(Exception):
        pass

    class WrapperB(Exception):
        pass

    wrapper_a = WrapperA("wrapper a")
    wrapper_a.__cause__ = KeyError("same inner type")
    wrapper_b = WrapperB("wrapper b")
    wrapper_b.__cause__ = KeyError("same inner type")
    # Both unwrap to KeyError, so they should match
    assert comparator(wrapper_a, wrapper_b)

    # Different wrapper types with different inner types should not match
    wrapper_c = WrapperA("wrapper c")
    wrapper_c.__cause__ = ValueError("value error")
    wrapper_d = WrapperB("wrapper d")
    wrapper_d.__cause__ = TypeError("type error")
    assert not comparator(wrapper_c, wrapper_d)


def test_comparator_with_message_pattern():
    """Test comparator with exception type extracted from message pattern."""
    # Exception with wrapped type in message (no __cause__)
    wrapper = RuntimeError("Operation failed: got IndexError('list index out of range')")

    plain_index = IndexError("some index error")
    assert comparator(wrapper, plain_index)
    assert comparator(plain_index, wrapper)

    # Should not match different types
    plain_key = KeyError("some key error")
    assert not comparator(wrapper, plain_key)


def test_comparator_wrapped_exceptions_bidirectional():
    """Test that wrapped exception comparison works in both directions."""

    class CustomWrapper(Exception):
        pass

    # Create wrapper with __cause__
    inner = AttributeError("attr error")
    wrapper = CustomWrapper("wrapper message")
    wrapper.__cause__ = inner

    plain_attr = AttributeError("plain attr error")

    # Test both directions
    assert comparator(wrapper, plain_attr)
    assert comparator(plain_attr, wrapper)

    # Test with superset_obj flag
    assert comparator(wrapper, plain_attr, superset_obj=True)
    assert comparator(plain_attr, wrapper, superset_obj=True)


def test_comparator_same_type_exceptions_still_work():
    """Ensure that same-type exception comparison still works correctly."""
    exc1 = ValueError("message 1")
    exc2 = ValueError("message 2")
    assert comparator(exc1, exc2)

    # With custom attributes
    class CustomError(Exception):
        def __init__(self, msg, code):
            super().__init__(msg)
            self.code = code

    custom1 = CustomError("msg1", 100)
    custom2 = CustomError("msg2", 100)
    assert comparator(custom1, custom2)

    custom3 = CustomError("msg3", 200)
    assert not comparator(custom1, custom3)


def test_comparator_no_false_positives_for_wrapped_exceptions():
    """Test that unrelated exception types don't match due to wrapping logic."""
    # Two completely different exception types should never match
    val_err = ValueError("value error")
    type_err = TypeError("type error")
    assert not comparator(val_err, type_err)

    # Wrapper with different inner type should not match
    wrapper = RuntimeError("some error")
    wrapper.__cause__ = KeyError("key error")
    assert not comparator(wrapper, val_err)
    assert not comparator(val_err, wrapper)


def test_collections() -> None:
    # Deque
    a = deque([1, 2, 3])
    b = deque([1, 2, 3])
    c = deque([1, 2, 4])
    d = deque([1, 2])
    e = [1, 2, 3]
    f = deque([1, 2, 3], maxlen=5)
    assert comparator(a, b)
    assert comparator(a, f)  # same elements, different maxlen is ok
    assert not comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(a, e)

    g = deque([{"a": 1}, {"b": 2}])
    h = deque([{"a": 1}, {"b": 2}])
    i = deque([{"a": 1}, {"b": 3}])
    assert comparator(g, h)
    assert not comparator(g, i)

    empty_deque1 = deque()
    empty_deque2 = deque()
    assert comparator(empty_deque1, empty_deque2)
    assert not comparator(empty_deque1, a)

    # namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    a = Point(x=1, y=2)
    b = Point(x=1, y=2)
    c = Point(x=1, y=3)
    assert comparator(a, b)
    assert not comparator(a, c)

    Point2 = namedtuple('Point2', ['x', 'y'])
    d = Point2(x=1, y=2)
    assert not comparator(a, d)

    e = (1, 2)
    assert not comparator(a, e)

    # ChainMap
    map1 = {'a': 1, 'b': 2}
    map2 = {'c': 3, 'd': 4}
    a = ChainMap(map1, map2)
    b = ChainMap(map1, map2)
    c = ChainMap(map2, map1)
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # Counter
    a = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
    b = Counter({'a': 3, 'b': 2, 'c': 1})
    c = Counter({'a': 3, 'b': 2, 'c': 2})
    d = {'a': 3, 'b': 2, 'c': 1}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # OrderedDict
    a = OrderedDict([('a', 1), ('b', 2)])
    b = OrderedDict([('a', 1), ('b', 2)])
    c = OrderedDict([('b', 2), ('a', 1)])
    d = {'a': 1, 'b': 2}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # defaultdict
    a = defaultdict(int, {'a': 1, 'b': 2})
    b = defaultdict(int, {'a': 1, 'b': 2})
    c = defaultdict(list, {'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    e = defaultdict(int, {'a': 1, 'b': 3})
    assert comparator(a, b)
    assert comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(a, e)

    # UserDict
    a = UserDict({'a': 1, 'b': 2})
    b = UserDict({'a': 1, 'b': 2})
    c = UserDict({'a': 1, 'b': 3})
    d = {'a': 1, 'b': 2}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # UserList
    a = UserList([1, 2, 3])
    b = UserList([1, 2, 3])
    c = UserList([1, 2, 4])
    d = [1, 2, 3]
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    # UserString
    a = UserString("hello")
    b = UserString("hello")
    c = UserString("world")
    d = "hello"
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)


def test_attrs():
    try:
        import attrs  # type: ignore
    except ImportError:
        pytest.skip()

    @attrs.define
    class Person:
        name: str
        age: int = 10
        
    a = Person("Alice", 25)
    b = Person("Alice", 25)
    c = Person("Bob", 25)
    d = Person("Alice", 30)
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    @attrs.frozen
    class Point:
        x: int
        y: int
        
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    p3 = Point(2, 3)
    assert comparator(p1, p2)
    assert not comparator(p1, p3)

    @attrs.define(slots=True)
    class Vehicle:
        brand: str
        model: str
        year: int = 2020
        
    v1 = Vehicle("Toyota", "Camry", 2021)
    v2 = Vehicle("Toyota", "Camry", 2021)
    v3 = Vehicle("Honda", "Civic", 2021)
    assert comparator(v1, v2)
    assert not comparator(v1, v3)

    @attrs.define
    class ComplexClass:
        public_field: str
        private_field: str = attrs.field(repr=False)
        non_eq_field: int = attrs.field(eq=False, default=0)
        computed: str = attrs.field(init=False, eq=True)
        
        def __attrs_post_init__(self):
            self.computed = f"{self.public_field}_{self.private_field}"
    
    c1 = ComplexClass("test", "secret")
    c2 = ComplexClass("test", "secret")
    c3 = ComplexClass("different", "secret")
    
    c1.non_eq_field = 100
    c2.non_eq_field = 200
    
    assert comparator(c1, c2)
    assert not comparator(c1, c3)

    @attrs.define
    class Address:
        street: str
        city: str
        
    @attrs.define 
    class PersonWithAddress:
        name: str
        address: Address
        
    addr1 = Address("123 Main St", "Anytown")
    addr2 = Address("123 Main St", "Anytown")
    addr3 = Address("456 Oak Ave", "Anytown")
    
    person1 = PersonWithAddress("John", addr1)
    person2 = PersonWithAddress("John", addr2)
    person3 = PersonWithAddress("John", addr3)
    
    assert comparator(person1, person2)
    assert not comparator(person1, person3)

    @attrs.define
    class Container:
        items: list
        metadata: dict
        
    cont1 = Container([1, 2, 3], {"type": "numbers"})
    cont2 = Container([1, 2, 3], {"type": "numbers"})
    cont3 = Container([1, 2, 4], {"type": "numbers"})
    
    assert comparator(cont1, cont2)
    assert not comparator(cont1, cont3)

    @attrs.define
    class BaseClass:
        name: str
        value: int
        
    @attrs.define
    class ExtendedClass:
        name: str
        value: int
        extra_field: str = "default"
        
    base = BaseClass("test", 42)
    extended = ExtendedClass("test", 42, "extra")
    
    assert not comparator(base, extended)

    @attrs.define
    class WithNonEqFields:
        name: str
        timestamp: float = attrs.field(eq=False)  # Should be ignored
        debug_info: str = attrs.field(eq=False, default="debug")
        
    obj1 = WithNonEqFields("test", 1000.0, "info1")
    obj2 = WithNonEqFields("test", 9999.0, "info2")  # Different non-eq fields
    obj3 = WithNonEqFields("different", 1000.0, "info1")
    
    assert comparator(obj1, obj2)  # Should be equal despite different timestamp/debug_info
    assert not comparator(obj1, obj3)  # Should be different due to name
    @attrs.define
    class MinimalClass:
        name: str
        value: int
        
    @attrs.define
    class ExtendedClass:
        name: str
        value: int
        extra_field: str = "default"
        metadata: dict = attrs.field(factory=dict)
        timestamp: float = attrs.field(eq=False, default=0.0)  # This should be ignored
        
    minimal = MinimalClass("test", 42)
    extended = ExtendedClass("test", 42, "extra", {"key": "value"}, 1000.0)

    assert not comparator(minimal, extended)


def test_dict_views() -> None:
    """Test comparator support for dict_keys, dict_values, and dict_items."""
    # Test dict_keys
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"a": 1, "b": 2, "c": 3}
    d3 = {"a": 1, "b": 2, "d": 3}
    d4 = {"a": 1, "b": 2}

    # dict_keys - same keys
    assert comparator(d1.keys(), d2.keys())
    # dict_keys - different keys
    assert not comparator(d1.keys(), d3.keys())
    # dict_keys - different length
    assert not comparator(d1.keys(), d4.keys())

    # Test dict_values
    v1 = {"a": 1, "b": 2, "c": 3}
    v2 = {"x": 1, "y": 2, "z": 3}  # same values, different keys
    v3 = {"a": 1, "b": 2, "c": 4}  # different value
    v4 = {"a": 1, "b": 2}  # different length

    # dict_values - same values (order matters for values since they're iterable)
    assert comparator(v1.values(), v2.values())
    # dict_values - different values
    assert not comparator(v1.values(), v3.values())
    # dict_values - different length
    assert not comparator(v1.values(), v4.values())

    # Test dict_items
    i1 = {"a": 1, "b": 2, "c": 3}
    i2 = {"a": 1, "b": 2, "c": 3}
    i3 = {"a": 1, "b": 2, "c": 4}  # different value
    i4 = {"a": 1, "b": 2, "d": 3}  # different key
    i5 = {"a": 1, "b": 2}  # different length
    i6 = {"b": 2, "c": 3, "a": 1}  # different order

    # dict_items - same items
    assert comparator(i1.items(), i2.items())
    # dict_items - different value
    assert not comparator(i1.items(), i3.items())
    # dict_items - different key
    assert not comparator(i1.items(), i4.items())
    # dict_items - different length
    assert not comparator(i1.items(), i5.items())

    assert comparator(i1.items(), i6.items())

    # Test empty dicts
    empty1 = {}
    empty2 = {}
    assert comparator(empty1.keys(), empty2.keys())
    assert comparator(empty1.values(), empty2.values())
    assert comparator(empty1.items(), empty2.items())

    # Test with nested values
    nested1 = {"a": [1, 2, 3], "b": {"x": 1}}
    nested2 = {"a": [1, 2, 3], "b": {"x": 1}}
    nested3 = {"a": [1, 2, 4], "b": {"x": 1}}

    assert comparator(nested1.values(), nested2.values())
    assert not comparator(nested1.values(), nested3.values())
    assert comparator(nested1.items(), nested2.items())
    assert not comparator(nested1.items(), nested3.items())

    # Test that dict views are not equal to lists/sets
    d = {"a": 1, "b": 2}
    assert not comparator(d.keys(), ["a", "b"])
    assert not comparator(d.keys(), {"a", "b"})
    assert not comparator(d.values(), [1, 2])
    assert not comparator(d.items(), [("a", 1), ("b", 2)])


def test_tensorflow_tensor() -> None:
    """Test comparator support for TensorFlow Tensor objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test basic 1D tensors
    a = tf.constant([1, 2, 3])
    b = tf.constant([1, 2, 3])
    c = tf.constant([1, 2, 4])

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test 2D tensors
    d = tf.constant([[1, 2, 3], [4, 5, 6]])
    e = tf.constant([[1, 2, 3], [4, 5, 6]])
    f = tf.constant([[1, 2, 3], [4, 5, 7]])

    assert comparator(d, e)
    assert not comparator(d, f)

    # Test tensors with different shapes
    g = tf.constant([1, 2, 3])
    h = tf.constant([[1, 2, 3]])

    assert not comparator(g, h)

    # Test tensors with different dtypes
    i = tf.constant([1, 2, 3], dtype=tf.float32)
    j = tf.constant([1, 2, 3], dtype=tf.float32)
    k = tf.constant([1, 2, 3], dtype=tf.int32)

    assert comparator(i, j)
    assert not comparator(i, k)

    # Test 3D tensors
    l = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    m = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    n = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 9]]])

    assert comparator(l, m)
    assert not comparator(l, n)

    # Test empty tensors
    o = tf.constant([])
    p = tf.constant([])
    q = tf.constant([1.0])

    assert comparator(o, p)
    assert not comparator(o, q)

    # Test tensors with NaN values
    r = tf.constant([1.0, float('nan'), 3.0])
    s = tf.constant([1.0, float('nan'), 3.0])
    t = tf.constant([1.0, 2.0, 3.0])

    assert comparator(r, s)  # NaN == NaN should be True
    assert not comparator(r, t)

    # Test tensors with infinity values
    u = tf.constant([1.0, float('inf'), 3.0])
    v = tf.constant([1.0, float('inf'), 3.0])
    w = tf.constant([1.0, float('-inf'), 3.0])

    assert comparator(u, v)
    assert not comparator(u, w)

    # Test complex tensors
    x = tf.constant([1+2j, 3+4j])
    y = tf.constant([1+2j, 3+4j])
    z = tf.constant([1+2j, 3+5j])

    assert comparator(x, y)
    assert not comparator(x, z)

    # Test boolean tensors
    aa = tf.constant([True, False, True])
    bb = tf.constant([True, False, True])
    cc = tf.constant([True, True, True])

    assert comparator(aa, bb)
    assert not comparator(aa, cc)

    # Test string tensors
    dd = tf.constant(["hello", "world"])
    ee = tf.constant(["hello", "world"])
    ff = tf.constant(["hello", "there"])

    assert comparator(dd, ee)
    assert not comparator(dd, ff)


def test_tensorflow_dtype() -> None:
    """Test comparator support for TensorFlow DType objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test float dtypes
    a = tf.float32
    b = tf.float32
    c = tf.float64

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test integer dtypes
    d = tf.int32
    e = tf.int32
    f = tf.int64

    assert comparator(d, e)
    assert not comparator(d, f)

    # Test unsigned integer dtypes
    g = tf.uint8
    h = tf.uint8
    i = tf.uint16

    assert comparator(g, h)
    assert not comparator(g, i)

    # Test complex dtypes
    j = tf.complex64
    k = tf.complex64
    l = tf.complex128

    assert comparator(j, k)
    assert not comparator(j, l)

    # Test bool dtype
    m = tf.bool
    n = tf.bool
    o = tf.int8

    assert comparator(m, n)
    assert not comparator(m, o)

    # Test string dtype
    p = tf.string
    q = tf.string
    r = tf.int32

    assert comparator(p, q)
    assert not comparator(p, r)


def test_tensorflow_variable() -> None:
    """Test comparator support for TensorFlow Variable objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test basic variables
    a = tf.Variable([1, 2, 3], dtype=tf.float32)
    b = tf.Variable([1, 2, 3], dtype=tf.float32)
    c = tf.Variable([1, 2, 4], dtype=tf.float32)

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test variables with different dtypes
    d = tf.Variable([1, 2, 3], dtype=tf.float32)
    e = tf.Variable([1, 2, 3], dtype=tf.float64)

    assert not comparator(d, e)

    # Test 2D variables
    f = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
    g = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
    h = tf.Variable([[1, 2], [3, 5]], dtype=tf.float32)

    assert comparator(f, g)
    assert not comparator(f, h)

    # Test variables with different shapes
    i = tf.Variable([1, 2, 3], dtype=tf.float32)
    j = tf.Variable([[1, 2, 3]], dtype=tf.float32)

    assert not comparator(i, j)


def test_tensorflow_tensor_shape() -> None:
    """Test comparator support for TensorFlow TensorShape objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test equal shapes
    a = tf.TensorShape([2, 3, 4])
    b = tf.TensorShape([2, 3, 4])
    c = tf.TensorShape([2, 3, 5])

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test different ranks
    d = tf.TensorShape([2, 3])
    e = tf.TensorShape([2, 3, 4])

    assert not comparator(d, e)

    # Test scalar shapes
    f = tf.TensorShape([])
    g = tf.TensorShape([])
    h = tf.TensorShape([1])

    assert comparator(f, g)
    assert not comparator(f, h)

    # Test shapes with None dimensions (unknown dimensions)
    i = tf.TensorShape([None, 3, 4])
    j = tf.TensorShape([None, 3, 4])
    k = tf.TensorShape([2, 3, 4])

    assert comparator(i, j)
    assert not comparator(i, k)

    # Test fully unknown shapes
    l = tf.TensorShape(None)
    m = tf.TensorShape(None)
    n = tf.TensorShape([1, 2])

    assert comparator(l, m)
    assert not comparator(l, n)


def test_tensorflow_sparse_tensor() -> None:
    """Test comparator support for TensorFlow SparseTensor objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test equal sparse tensors
    a = tf.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1.0, 2.0],
        dense_shape=[3, 4]
    )
    b = tf.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1.0, 2.0],
        dense_shape=[3, 4]
    )
    c = tf.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1.0, 3.0],  # Different value
        dense_shape=[3, 4]
    )

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test sparse tensors with different indices
    d = tf.SparseTensor(
        indices=[[0, 0], [1, 3]],  # Different index
        values=[1.0, 2.0],
        dense_shape=[3, 4]
    )

    assert not comparator(a, d)

    # Test sparse tensors with different shapes
    e = tf.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[1.0, 2.0],
        dense_shape=[4, 5]  # Different shape
    )

    assert not comparator(a, e)

    # Test empty sparse tensors
    f = tf.SparseTensor(
        indices=tf.zeros([0, 2], dtype=tf.int64),
        values=[],
        dense_shape=[3, 4]
    )
    g = tf.SparseTensor(
        indices=tf.zeros([0, 2], dtype=tf.int64),
        values=[],
        dense_shape=[3, 4]
    )

    assert comparator(f, g)


def test_tensorflow_ragged_tensor() -> None:
    """Test comparator support for TensorFlow RaggedTensor objects."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow required for this test")

    # Test equal ragged tensors
    a = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
    b = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
    c = tf.ragged.constant([[1, 2], [3, 4, 6], [6]])  # Different value

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test ragged tensors with different row lengths
    d = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])  # Different structure

    assert not comparator(a, d)

    # Test ragged tensors with different dtypes
    e = tf.ragged.constant([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
    f = tf.ragged.constant([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])

    assert comparator(e, f)
    assert not comparator(a, e)  # int vs float

    # Test nested ragged tensors
    g = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]])
    h = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]])
    i = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 7]]])

    assert comparator(g, h)
    assert not comparator(g, i)

    # Test empty ragged tensors
    j = tf.ragged.constant([[], [], []])
    k = tf.ragged.constant([[], [], []])

    assert comparator(j, k)


def test_slice() -> None:
    """Test comparator support for slice objects."""
    # Test equal slices
    a = slice(1, 10, 2)
    b = slice(1, 10, 2)
    assert comparator(a, b)

    # Test slices with different start
    c = slice(2, 10, 2)
    assert not comparator(a, c)

    # Test slices with different stop
    d = slice(1, 11, 2)
    assert not comparator(a, d)

    # Test slices with different step
    e = slice(1, 10, 3)
    assert not comparator(a, e)

    # Test slices with None values
    f = slice(None, 10, 2)
    g = slice(None, 10, 2)
    h = slice(1, 10, 2)
    assert comparator(f, g)
    assert not comparator(f, h)

    # Test slices with all None (equivalent to [:])
    i = slice(None, None, None)
    j = slice(None, None, None)
    k = slice(None, None, 1)
    assert comparator(i, j)
    assert not comparator(i, k)

    # Test slices with only stop
    l = slice(5)
    m = slice(5)
    n = slice(6)
    assert comparator(l, m)
    assert not comparator(l, n)

    # Test slices with negative values
    o = slice(-5, -1, 1)
    p = slice(-5, -1, 1)
    q = slice(-5, -2, 1)
    assert comparator(o, p)
    assert not comparator(o, q)

    # Test slice is not equal to other types
    r = slice(1, 10)
    s = (1, 10)
    assert not comparator(r, s)


def test_numpy_datetime64() -> None:
    """Test comparator support for numpy datetime64 and timedelta64 types."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy required for this test")

    # Test datetime64 equality
    a = np.datetime64('2021-01-01')
    b = np.datetime64('2021-01-01')
    c = np.datetime64('2021-01-02')

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test datetime64 with different units
    d = np.datetime64('2021-01-01', 'D')
    e = np.datetime64('2021-01-01', 'D')
    f = np.datetime64('2021-01-01', 's')  # Different unit (seconds)

    assert comparator(d, e)
    # Note: datetime64 with different units but same moment may or may not be equal
    # depending on numpy version behavior

    # Test datetime64 with time
    g = np.datetime64('2021-01-01T12:00:00')
    h = np.datetime64('2021-01-01T12:00:00')
    i = np.datetime64('2021-01-01T12:00:01')

    assert comparator(g, h)
    assert not comparator(g, i)

    # Test timedelta64 equality
    j = np.timedelta64(1, 'D')
    k = np.timedelta64(1, 'D')
    l = np.timedelta64(2, 'D')

    assert comparator(j, k)
    assert not comparator(j, l)

    # Test timedelta64 with different units
    m = np.timedelta64(1, 'h')
    n = np.timedelta64(1, 'h')
    o = np.timedelta64(60, 'm')  # Same duration, different unit

    assert comparator(m, n)
    # 1 hour == 60 minutes, but they have different units
    # numpy may treat them as equal or not depending on comparison

    # Test NaT (Not a Time) - numpy's equivalent of NaN for datetime
    p = np.datetime64('NaT')
    q = np.datetime64('NaT')
    r = np.datetime64('2021-01-01')

    assert comparator(p, q)  # NaT == NaT should be True
    assert not comparator(p, r)

    # Test timedelta64 NaT
    s = np.timedelta64('NaT')
    t = np.timedelta64('NaT')
    u = np.timedelta64(1, 'D')

    assert comparator(s, t)  # NaT == NaT should be True
    assert not comparator(s, u)

    # Test datetime64 is not equal to other types
    v = np.datetime64('2021-01-01')
    w = '2021-01-01'
    assert not comparator(v, w)

    # Test arrays of datetime64
    x = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64')
    y = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64')
    z = np.array(['2021-01-01', '2021-01-03'], dtype='datetime64')

    assert comparator(x, y)
    assert not comparator(x, z)


def test_numpy_0d_array() -> None:
    """Test comparator handles 0-d numpy arrays without 'iteration over 0-d array' error."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy required for this test")

    # Test 0-d integer array
    a = np.array(5)
    b = np.array(5)
    c = np.array(6)

    assert comparator(a, b)
    assert not comparator(a, c)

    # Test 0-d float array
    d = np.array(3.14)
    e = np.array(3.14)
    f = np.array(2.71)

    assert comparator(d, e)
    assert not comparator(d, f)

    # Test 0-d complex array
    g = np.array(1+2j)
    h = np.array(1+2j)
    i = np.array(1+3j)

    assert comparator(g, h)
    assert not comparator(g, i)

    # Test 0-d string array
    j = np.array('hello')
    k = np.array('hello')
    l = np.array('world')

    assert comparator(j, k)
    assert not comparator(j, l)

    # Test 0-d boolean array
    m = np.array(True)
    n = np.array(True)
    o = np.array(False)

    assert comparator(m, n)
    assert not comparator(m, o)

    # Test 0-d array with NaN
    p = np.array(np.nan)
    q = np.array(np.nan)
    r = np.array(1.0)

    assert comparator(p, q)  # NaN == NaN should be True
    assert not comparator(p, r)

    # Test 0-d datetime64 array
    s = np.array(np.datetime64('2021-01-01'))
    t = np.array(np.datetime64('2021-01-01'))
    u = np.array(np.datetime64('2021-01-02'))

    assert comparator(s, t)
    assert not comparator(s, u)

    # Test 0-d array vs scalar
    v = np.array(5)
    w = 5
    # 0-d array and scalar are different types
    assert not comparator(v, w)

    # Test 0-d array vs 1-d array with one element
    x = np.array(5)
    y = np.array([5])
    # Different shapes
    assert not comparator(x, y)

def test_numpy_dtypes() -> None:
    """Test comparator for numpy.dtypes types like Float64DType, Int64DType, etc."""
    try:
        import numpy as np
        import numpy.dtypes as dtypes
    except ImportError:
        pytest.skip("numpy not available")

    # Test Float64DType
    a = dtypes.Float64DType()
    b = dtypes.Float64DType()
    assert comparator(a, b)

    # Test Int64DType
    c = dtypes.Int64DType()
    d = dtypes.Int64DType()
    assert comparator(c, d)

    # Test different DType classes should not be equal
    assert not comparator(a, c)  # Float64DType vs Int64DType

    # Test various numeric DType classes
    assert comparator(dtypes.Int8DType(), dtypes.Int8DType())
    assert comparator(dtypes.Int16DType(), dtypes.Int16DType())
    assert comparator(dtypes.Int32DType(), dtypes.Int32DType())
    assert comparator(dtypes.UInt8DType(), dtypes.UInt8DType())
    assert comparator(dtypes.UInt16DType(), dtypes.UInt16DType())
    assert comparator(dtypes.UInt32DType(), dtypes.UInt32DType())
    assert comparator(dtypes.UInt64DType(), dtypes.UInt64DType())
    assert comparator(dtypes.Float32DType(), dtypes.Float32DType())
    assert comparator(dtypes.Complex64DType(), dtypes.Complex64DType())
    assert comparator(dtypes.Complex128DType(), dtypes.Complex128DType())
    assert comparator(dtypes.BoolDType(), dtypes.BoolDType())

    # Test cross-type comparisons should be False
    assert not comparator(dtypes.Int32DType(), dtypes.Int64DType())
    assert not comparator(dtypes.Float32DType(), dtypes.Float64DType())
    assert not comparator(dtypes.UInt32DType(), dtypes.Int32DType())

    # Test regular np.dtype instances
    e = np.dtype('float64')
    f = np.dtype('float64')
    assert comparator(e, f)

    g = np.dtype('int64')
    h = np.dtype('int64')
    assert comparator(g, h)

    assert not comparator(e, g)  # float64 vs int64

    # Test DType class instances vs regular np.dtype (they should be equal if same underlying type)
    assert comparator(dtypes.Float64DType(), np.dtype('float64'))
    assert comparator(dtypes.Int64DType(), np.dtype('int64'))
    assert comparator(dtypes.Int32DType(), np.dtype('int32'))
    assert comparator(dtypes.BoolDType(), np.dtype('bool'))

    # Test that DType and np.dtype of different types are not equal
    assert not comparator(dtypes.Float64DType(), np.dtype('int64'))
    assert not comparator(dtypes.Int32DType(), np.dtype('float32'))


def test_numpy_extended_precision_types() -> None:
    """Test comparator for numpy extended precision types like clongdouble."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")

    # Test np.clongdouble (extended precision complex)
    c1 = np.clongdouble(1 + 2j)
    c2 = np.clongdouble(1 + 2j)
    c3 = np.clongdouble(1 + 3j)
    assert comparator(c1, c2)
    assert not comparator(c1, c3)

    # Test np.longdouble (extended precision float)
    l1 = np.longdouble(1.5)
    l2 = np.longdouble(1.5)
    l3 = np.longdouble(2.5)
    assert comparator(l1, l2)
    assert not comparator(l1, l3)

    # Test NaN handling for extended precision complex
    nan_c1 = np.clongdouble(complex(np.nan, 2))
    nan_c2 = np.clongdouble(complex(np.nan, 2))
    assert comparator(nan_c1, nan_c2)

    # Test NaN handling for extended precision float
    nan_l1 = np.longdouble(np.nan)
    nan_l2 = np.longdouble(np.nan)
    assert comparator(nan_l1, nan_l2)


def test_numpy_typing_types() -> None:
    """Test comparator for numpy.typing types like NDArray type aliases."""
    try:
        import numpy as np
        import numpy.typing as npt
    except ImportError:
        pytest.skip("numpy or numpy.typing not available")

    # Test NDArray type alias comparisons
    arr_type1 = npt.NDArray[np.float64]
    arr_type2 = npt.NDArray[np.float64]
    arr_type3 = npt.NDArray[np.int64]
    assert comparator(arr_type1, arr_type2)
    assert not comparator(arr_type1, arr_type3)

    # Test NBitBase (if it can be instantiated)
    try:
        nbit1 = npt.NBitBase()
        nbit2 = npt.NBitBase()
        # NBitBase instances with empty __dict__ should compare as equal
        assert comparator(nbit1, nbit2)
        # Also test with superset_obj=True
        assert comparator(nbit1, nbit2, superset_obj=True)
    except TypeError:
        # NBitBase may not be instantiable in all numpy versions
        pass


def test_numpy_typing_superset_obj() -> None:
    """Test comparator with superset_obj=True for numpy types."""
    try:
        import numpy as np
        import numpy.typing as npt
    except ImportError:
        pytest.skip("numpy or numpy.typing not available")

    # Test numpy arrays with object dtype containing dicts (superset scenario)
    a1 = np.array([{'a': 1}], dtype=object)
    a2 = np.array([{'a': 1, 'b': 2}], dtype=object)  # superset
    assert comparator(a1, a2, superset_obj=True)
    assert not comparator(a1, a2, superset_obj=False)

    # Test extended precision types with superset_obj=True
    c1 = np.clongdouble(1 + 2j)
    c2 = np.clongdouble(1 + 2j)
    assert comparator(c1, c2, superset_obj=True)

    l1 = np.longdouble(1.5)
    l2 = np.longdouble(1.5)
    assert comparator(l1, l2, superset_obj=True)

    # Test NDArray type alias with superset_obj=True
    arr_type1 = npt.NDArray[np.float64]
    arr_type2 = npt.NDArray[np.float64]
    assert comparator(arr_type1, arr_type2, superset_obj=True)

    # Test numpy structured arrays (np.void) with superset_obj=True
    dt = np.dtype([('name', 'S10'), ('age', np.int32)])
    a_struct = np.array([('Alice', 25)], dtype=dt)
    b_struct = np.array([('Alice', 25)], dtype=dt)
    assert comparator(a_struct[0], b_struct[0], superset_obj=True)

    # Test numpy random generators with superset_obj=True
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    assert comparator(rng1, rng2, superset_obj=True)

    rs1 = np.random.RandomState(seed=42)
    rs2 = np.random.RandomState(seed=42)
    assert comparator(rs1, rs2, superset_obj=True)
