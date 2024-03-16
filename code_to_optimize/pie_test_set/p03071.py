def problem_p03071(input_data):
    import heapq

    a, b = list(map(int, input_data.split()))

    return sum(heapq.nlargest(2, [a, a - 1, b, b - 1]))
