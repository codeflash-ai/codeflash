def problem_p02989():
    N = int(eval(input()))

    d = list(map(int, input().split()))

    d.sort()

    harf_N = int(N / 2)

    print((d[harf_N] - d[harf_N - 1]))


problem_p02989()
