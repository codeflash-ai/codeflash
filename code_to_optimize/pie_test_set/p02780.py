def problem_p02780():
    import numpy as np

    n, k = list(map(int, input().split()))

    exp = tuple([(int(x) + 1) / 2 for x in input().split()])

    l = np.cumsum(exp)

    ans = l[k - 1]

    for i in range(0, n - k):

        x = l[i + k] - l[i]

        ans = max(ans, x)

    print(ans)


problem_p02780()
