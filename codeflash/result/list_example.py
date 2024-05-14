from collections import defaultdict
from typing import List, Set, Tuple
import itertools
from functools import reduce
from typing import List, Set, Tuple


def compare_lists(
    li1: List[int],
    li2: List[int],
    value_func1=None,
    value_func2=None,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Compare *li1* and *li2*, return the results as a list in the following form:

    [[data seen in both lists], [data only seen in li1], [data only seen in li2]]

    and [data seen in both lists] contains 2 tuple: [(actual items in li1), (actual items in li2)]

    * *value_func1* callback function to li1, applied to each item in the list, returning the **logical** value for comparison
    * *value_func2* callback function to li2, similarly

    If not supplied, lists will be compared as it is.

    Usage::

        >>> compare_lists([1, 2, 3], [1, 3, 5])
        >>> ([(1, 3), (1, 3)], [2], [5])

    Or with callback functions specified::

        >>> f = lambda x: x['v']
        >>>
        >>> li1 = [{'v': 1}, {'v': 2}, {'v': 3}]
        >>> li2 = [1, 3, 5]
        >>>
        >>> compare_lists(li1, li2, value_func1=f)
        >>> ([({'v': 1}, {'v': 3}), (1, 3)], [{'v': 2}], [5])

    """

    if not value_func1:
        value_func1 = lambda x: x
    if not value_func2:
        value_func2 = lambda x: x

    def to_dict(li, vfunc):
        d = defaultdict(list)
        for item in li:
            d[vfunc(item)].append(item)
        return d

    def flatten(li):
        return [elem for sublist in li for elem in sublist]

    d1 = to_dict(li1, value_func1)
    d2 = to_dict(li2, value_func2)

    if d1 == d2:
        return (li1, li2), [], []

    k1, k2 = set(d1.keys()), set(d2.keys())

    elems_left = flatten([d1[k] for k in k1 - k2])
    elems_right = flatten([d2[k] for k in k2 - k1])
    elems_both = flatten([d2[k] for k in k1 & k2])

    return set(elems_both), set(elems_left), set(elems_right)
