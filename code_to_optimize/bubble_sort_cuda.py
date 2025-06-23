import torch

def sorter_cuda(arr: torch.Tensor)->torch.Tensor:
    arr = arr.cuda()
    print("codeflash stdout: Sorting list")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0] - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    print(f"result: {arr}")
    return arr.cpu()
