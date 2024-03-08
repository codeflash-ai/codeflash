def problem_p00070():
    import sys

    def f0070(A, n, s):

        if n == 1:
            return s in A

        A = sorted(A)

        b1 = 0

        b2 = 0

        c = 0

        j = 1

        for e, f in zip(A[:n][::-1], A[-n:]):

            b1 += j * e

            b2 += j * f

            j += 1

        if b1 <= s <= b2:

            for i in range(len(A)):

                c += f0070(A[:i] + A[i + 1 :], n - 1, s - A[i] * n)

        return c

    for a in sys.stdin:

        n, s = list(map(int, a.split()))

        print(f0070(list(range(10)), n, s))


problem_p00070()
