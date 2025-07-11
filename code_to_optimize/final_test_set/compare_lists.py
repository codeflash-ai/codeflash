from typing import List, Set, Tuple


def compare_lists(
    li1: List[int], li2: List[int], value_func1=None, value_func2=None
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
    if value_func1 is None:
        value_func1 = lambda x: x
    if value_func2 is None:
        value_func2 = lambda x: x

    # Build dict of value: [items] using a single loop each
    def to_dict(li, vfunc):
        d = {}
        for item in li:
            k = vfunc(item)
            if k in d:
                d[k].append(item)
            else:
                d[k] = [item]
        return d

    d1 = to_dict(li1, value_func1)
    d2 = to_dict(li2, value_func2)

    # Short-circuit for identical key/item dicts
    if d1 == d2:
        return set(li1), set(), set()

    k1 = set(d1.keys())
    k2 = set(d2.keys())

    only_in_li1 = []
    for k in k1 - k2:
        only_in_li1.extend(d1[k])

    only_in_li2 = []
    for k in k2 - k1:
        only_in_li2.extend(d2[k])

    in_both = []
    for k in k1 & k2:
        in_both.extend(d2[k])

    return set(in_both), set(only_in_li1), set(only_in_li2)
