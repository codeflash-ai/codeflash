def problem_p03614():
    # python template for atcoder1

    import sys

    sys.setrecursionlimit(10**9)

    input = sys.stdin.readline

    N = int(eval(input()))

    A = list(map(int, input().split()))

    seq = 0

    ans = 0

    for i in range(N):

        A[i] -= i + 1

        if A[i] == 0:

            seq += 1

        else:

            ans += (seq + 1) // 2

            seq = 0

    ans += (seq + 1) // 2

    print(ans)


problem_p03614()
