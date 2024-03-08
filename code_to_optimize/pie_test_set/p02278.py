def problem_p02278():
    """Minimum cost Sort."""

    def min_cost_sort(A):
        """Sort list A in ascending order.



        And return the switching cost in sorting.

        """

        B = list(A)

        B.sort()

        cost = 0

        min_w = B[0]

        for i, b in enumerate(B):

            tmp_cost = 0

            bi = A.index(b)

            n = 1

            while bi != i:

                n += 1

                st = B[bi]

                si = A.index(st)

                tmp_cost += b + st

                A[bi], A[si] = st, b

                bi = si

            dec = (n - 1) * (b - min_w)

            inc = 2 * (min_w + b)

            if dec < inc:

                cost += tmp_cost

            else:

                cost += tmp_cost - dec + inc

        return cost

    n = eval(input())

    A = list(map(int, input().split()))

    ans = min_cost_sort(A)

    print(ans)


problem_p02278()
