def problem_p02866():
    N = int(eval(input()))

    D = list(map(int, input().split()))

    M = 998244353

    from collections import Counter

    if D[0] != 0:

        print((0))

        exit(0)

    cd = Counter(D)

    if cd[0] != 1:

        print((0))

        exit(0)

    tmp = sorted(list(cd.items()), key=lambda x: x[0])

    ans = 1

    for kx in range(1, max(D) + 1):

        ans *= pow(cd[kx - 1], cd[kx], M)

        ans %= M

    print(ans)


problem_p02866()
