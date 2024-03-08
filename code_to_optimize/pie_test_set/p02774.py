def problem_p02774():
    import sys

    import bisect

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    in_n = lambda: int(readline())

    in_nn = lambda: list(map(int, readline().split()))

    in_nl = lambda: list(map(int, readline().split()))

    in_na = lambda: list(map(int, read().split()))

    in_s = lambda: readline().rstrip().decode("utf-8")

    INF = 10**18

    def binary_search(min_n, max_n, judge):

        while max_n - min_n != 1:

            tn = (min_n + max_n) // 2

            if judge(tn):

                max_n = tn

            else:

                min_n = tn

        return max_n

    def main():

        N, K = in_nn()

        A = in_nl()

        Ap = []

        Am = []

        Nz = 0

        for i in range(N):

            if A[i] > 0:

                Ap.append(A[i])

            elif A[i] == 0:

                Nz += 1

            else:

                Am.append(A[i])

        Np = len(Ap)

        Nm = len(Am)

        m_count = Np * Nm

        z_count = Nz * (Np + Nm) + Nz * (Nz - 1) // 2

        # p_count = Np * (Np - 1) // 2 + Nm * (Nm - 1) // 2

        if K <= m_count:

            Ap.sort()

            Am.sort()

            def judge(tn):

                count = 0

                for i in range(Nm):

                    search = -(-tn // Am[i])

                    j = bisect.bisect_left(Ap, search)

                    count += Np - j

                return count >= K

            ans = binary_search(-INF, -1, judge)

        elif K <= m_count + z_count:

            ans = 0

        else:

            Ap.sort()

            Am = sorted([-a for a in Am])

            def judge(tn):

                count = 0

                for i in range(Np):

                    search = tn // Ap[i]

                    j = bisect.bisect_right(Ap, search)

                    count += max(0, j - i - 1)

                for i in range(Nm):

                    search = tn // Am[i]

                    j = bisect.bisect_right(Am, search)

                    count += max(0, j - i - 1)

                return count >= (K - m_count - z_count)

            ans = binary_search(1, INF, judge)

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02774()
