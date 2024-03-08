def problem_p03667():
    import sys

    input = sys.stdin.readline

    """
    
    ・とりあえず部分点解法：各クエリに対してO(N)
    
    ・いくつかを残していくつかを自由に埋める
    
    ・残すもの：被覆区間がoverlapしないように残す
    
    ・つまり、区間で覆えている点が処理できて、区間で覆えていない点が他所から持ってこないといけない。
    
    """

    N, M = list(map(int, input().split()))

    A = [int(x) for x in input().split()]

    XY = [tuple(int(x) for x in input().split()) for _ in range(M)]

    def subscore_solution():

        from collections import Counter

        for x, y in XY:

            A[x - 1] = y

            covered = [False] * (N + N + 10)

            for key, cnt in list(Counter(A).items()):

                for i in range(cnt):

                    covered[max(0, key - i)] = True

            print((sum(not bl for bl in covered[1 : N + 1])))

    counter = [0] * (N + 1)

    covered = [0] * (N + N + 10)

    for a in A:

        counter[a] += 1

        covered[a - counter[a] + 1] += 1

    magic = sum(x == 0 for x in covered[1 : N + 1])

    for i, y in XY:

        x = A[i - 1]

        A[i - 1] = y

        rem = x - counter[x] + 1

        counter[x] -= 1

        counter[y] += 1

        add = y - counter[y] + 1

        covered[rem] -= 1

        if 1 <= rem <= N and covered[rem] == 0:

            magic += 1

        if 1 <= add <= N and covered[add] == 0:

            magic -= 1

        covered[add] += 1

        print(magic)


problem_p03667()
