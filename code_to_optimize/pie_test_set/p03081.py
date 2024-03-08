def problem_p03081():
    n, q = list(map(int, input().split()))
    s = "_" + eval(input()) + "_"
    l, r = 0, n + 1

    for m, h in [input().split() for i in range(q)][::-1]:
        L = h == "L"
        R = h == "R"
        l, r = [l, l - 1, l + 1][(m == s[l] and R) - (m == s[l + 1] and L)], [r, r - 1, r + 1][
            (m == s[r - 1] and R) - (m == s[r] and L)
        ]

    print((max(0, r - l - 1)))


problem_p03081()
