def problem_p03026():
    from scipy.sparse import *

    n, *t = list(map(int, open(0).read().split()))

    c = sorted(t[-n:])

    i = n

    for j in csgraph.depth_first_order(
        csr_matrix((t[: n - 1], (t[:-n:2], t[1:-n:2])), [n + 1] * 2), 1, 0, 0
    ):
        i -= 1
        t[j - 1] = c[i]

    print((sum(c[:-1]), *t[:n]))


problem_p03026()
