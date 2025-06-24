from typing import Union

import torch

def sorter_cuda(arr: list[float])->list[float]:
    arr1 = torch.randperm(10).cuda()
    print("codeflash stdout: Sorting list")
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[0] - 1):
            if arr1[j] > arr1[j + 1]:
                temp = arr1[j]
                arr1[j] = arr1[j + 1]
                arr1[j + 1] = temp
    print(f"result: {arr}")
    arr1 = arr1.cpu()
    arr.sort()
    return arr
