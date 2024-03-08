def problem_p03360():
    a, b, c = list(map(int, input().split()))

    k = int(eval(input()))

    print((a + b + c - max(a, b, c) + max(a, b, c) * (2**k)))


problem_p03360()
