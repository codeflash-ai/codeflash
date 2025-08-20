from typing import List, Union


def sorter(arr: Union[List[int], List[float]]) -> Union[List[int], List[float]]:
    for i in range(len(arr)):
        swapped = False
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
                swapped = True
        if not swapped:
            break
    return arr
