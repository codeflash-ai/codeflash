def find_top_k_elements(arr: list, k):
    if k <= 0:
        return []

    if k >= len(arr):
        result = []
        for num in arr:
            result.append(num)
        result.sort(reverse=True)
        return result

    top_k = []

    for num in arr:
        top_k.append(num)
        top_k.sort(reverse=True)
        if len(top_k) > k:
            top_k.pop()

    return top_k
