def problem_p00605():
    while True:

        n, k = list(map(int, input().split()))

        if n == k == 0:

            break

        s = [int(x) for x in input().split()]

        for _ in range(n):

            b = [int(x) for x in input().split()]

            for i in range(k):

                s[i] -= b[i]

        print(("Yes" if min(s) >= 0 else "No"))


problem_p00605()
