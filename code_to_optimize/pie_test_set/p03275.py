def problem_p03275():
    N = int(eval(input()))

    A = [int(_) for _ in input().split()]

    class BIT:

        # index 1~

        def __init__(self, n):

            self.size = n

            self.tree = [0] * (n + 1)

        def sum(self, i):

            s = 0

            while i > 0:

                s += self.tree[i]

                i -= i & -i

            return s

        def add(self, i, x):

            while i <= self.size:

                self.tree[i] += x

                i += i & -i

    M = N * (N + 1) // 2  # (N+1)C2

    sorted_A = sorted(A)

    inf = 0

    sup = max(A) + 1

    while sup - inf > 1:

        x = (sup + inf) // 2

        S = [0] * (N + 1)

        for i in range(N):

            if A[i] >= x:

                S[i + 1] = S[i] + 1

            else:

                S[i + 1] = S[i] - 1

        min_value = min(S)

        for i in range(N + 1):

            S[i] += abs(min_value) + 1

        forward = 0

        bit = BIT(max(S))

        for s in S:

            forward += bit.sum(s)

            bit.add(s, 1)

        if forward >= (M + 1) // 2:

            inf = x

        else:

            sup = x

    for a in reversed(sorted_A):

        if inf >= a:

            print(a)

            break


problem_p03275()
