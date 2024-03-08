def problem_p02537():
    d = [0] * 6**8

    n, k, *a = list(map(int, open(0).read().split() + d))

    for i in range(n):
        d[i] = max(d[j] for j in range(i - 99, i + 1) if abs(a[i] - a[j]) <= k) + 1

    print((max(d)))


problem_p02537()
