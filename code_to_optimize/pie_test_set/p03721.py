def problem_p03721():
    import numpy as np

    n, k = list(map(int, input().split()))

    z = np.zeros(10**5 + 1)

    for i in range(n):

        a, b = list(map(int, input().split()))

        z[a] += b

    a = 0

    for i in range(1, 10**5 + 1):

        if k <= z[i]:

            a = i

            break

        k -= z[i]

    print((int(a)))


problem_p03721()
