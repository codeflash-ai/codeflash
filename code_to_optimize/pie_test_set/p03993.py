def problem_p03993():
    n = int(eval(input()))
    a = list(map(int, input().split()))
    print((sum([0, 1][a[a[i] - 1] - 1 == i] for i in range(n)) // 2))


problem_p03993()
