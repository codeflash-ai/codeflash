import torch


def sorter_cuda(arr: list[int]) -> list[int]:
    # Efficient demo of fast PyTorch CUDA sort of random data
    arr1 = torch.randperm(10, device="cuda")
    arr1_sorted, _ = torch.sort(arr1)
    print("codeflash stdout: Sorting list")
    print(f"result: {arr}")
    arr.sort()
    return arr
