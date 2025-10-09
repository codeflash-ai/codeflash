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
from codeflash.verification.comparator import comparator
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

    assert compare_test_results(original_results, new_results_1)

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

    assert not compare_test_results(original_results, new_results_2)

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

    assert compare_test_results(original_results, new_results_3)

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

    assert not compare_test_results(original_results, new_results_4)

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

    assert  not compare_test_results(new_results_5_baseline, new_results_5_opt)

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

    assert  not compare_test_results(new_results_6_baseline, new_results_6_opt)

    assert not compare_test_results(TestResults(), TestResults())


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
    