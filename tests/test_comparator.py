import datetime
import pytest

from codeflash.verification.comparator import comparator
from codeflash.verification.equivalence import compare_results
from codeflash.verification.test_results import (
    TestResults,
    FunctionTestInvocation,
    InvocationId,
    TestType,
)


def test_basic_python_objects():
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


def test_standard_python_library_objects():
    a = datetime.datetime(2020, 2, 2, 2, 2, 2)
    b = datetime.datetime(2020, 2, 2, 2, 2, 2)
    c = datetime.datetime(2020, 2, 2, 2, 2, 3)
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.date(2020, 2, 2)
    b = datetime.date(2020, 2, 2)
    c = datetime.date(2020, 2, 3)
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.timedelta(days=1)
    b = datetime.timedelta(days=1)
    c = datetime.timedelta(days=2)
    assert comparator(a, b)
    assert not comparator(a, c)


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
    print(comparator(j, k))
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


def test_scipy():
    try:
        import scipy as sp
    except ImportError:
        pytest.skip()
    a = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    b = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    c = sp.sparse.csr_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(a, b)
    assert not comparator(a, c)

    d = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    e = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 5]])
    f = sp.sparse.csc_matrix([[1, 0, 0], [0, 0, 3], [4, 0, 6]])
    assert comparator(d, e)
    assert not comparator(d, f)
    assert not comparator(a, d)

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
        pass


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
        def __init__(self, value):
            self.value = value

    a = TestClass(5)
    b = TestClass2(5)
    c = TestClass2(5)
    assert not comparator(a, b)
    assert comparator(
        b, c
    )  # This is a fallback to True right now since we don't know how to compare them. This can be improved later

    class TestClass3(TestClass):
        def print(self):
            print(self.value)

    a = TestClass2(5)
    b = TestClass3(5)
    c = TestClass3(5)
    assert not comparator(a, b)
    assert comparator(b, c)


def test_compare_results_fn():
    original_results = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=5,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            )
        ]
    )

    new_results_1 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            )
        ]
    )

    assert compare_results(original_results, new_results_1)

    new_results_2 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=[5],
            )
        ]
    )

    assert not compare_results(original_results, new_results_2)

    new_results_3 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            ),
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="2",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            ),
        ]
    )

    assert not compare_results(original_results, new_results_3)

    assert not compare_results(TestResults(), TestResults())
