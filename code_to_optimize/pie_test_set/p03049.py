def problem_p03049():
    from collections import defaultdict, deque, Counter

    def inpl():
        return list(map(int, input().split()))

    N = int(eval(input()))

    A = 0

    BA = 0

    B = 0

    Z = 0

    ans = 0

    for _ in range(N):

        s = eval(input())

        ans += s.count("AB")

        if s[0] == "B":

            if s[-1] == "A":

                BA += 1

            else:

                B += 1

        elif s[-1] == "A":

            A += 1

    if BA:

        ans += BA - 1

        if A:

            A -= 1

            ans += 1

        if B:

            B -= 1

            ans += 1

    ans += min(A, B)

    print(ans)


problem_p03049()
