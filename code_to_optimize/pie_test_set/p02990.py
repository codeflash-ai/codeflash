def problem_p02990():
    from scipy.misc import *

    n, k = list(map(int, input().split()))

    for i in range(k):
        print((comb(n - k + 1, i + 1, 1) * comb(k - 1, i, 1) % (10**9 + 7)))


problem_p02990()
