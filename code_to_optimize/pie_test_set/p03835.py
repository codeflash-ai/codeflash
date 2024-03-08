def problem_p03835():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    K, S = list(map(int, input().split()))

    ans = 0

    for z in range(K + 1):

        for y in range(K + 1):

            x = S - (z + y)

            if x >= 0 and x <= K:

                ans += 1

    print(ans)


problem_p03835()
