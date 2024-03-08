def problem_p00506():
    input()

    n = list(map(int, input().split()))

    [print(x) for x in range(1, min(n) + 1) if sum(m % x for m in n) == 0]


problem_p00506()
