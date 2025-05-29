from typing import List, Union


def sorter(arr: Union[List[int], List[float]]) -> Union[List[int], List[float]]:
    # Use in-place sort for better performance
    arr.sort()
    return arr
