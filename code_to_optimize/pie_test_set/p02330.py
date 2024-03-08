def problem_p02330():
    import sys

    import bisect

    def main():

        n, K, L, R = list(map(int, sys.stdin.readline().split()))

        a = tuple(map(int, sys.stdin.readline().split()))

        m = n // 2

        ls = [[] for _ in range(m + 1)]

        for i in range(1 << m):

            ls[bin(i).count("1")].append(sum([a[j] for j in range(m) if i >> j & 1]))

        for i in range(m + 1):

            ls[i].sort()

        ans = 0

        for i in range(1 << n - m):

            cnt = bin(i).count("1")

            val = sum([a[m + j] for j in range(n - m) if i >> j & 1])

            if K - m <= cnt <= K:

                ans += bisect.bisect_right(ls[K - cnt], R - val) - bisect.bisect_right(
                    ls[K - cnt], L - val - 1
                )

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02330()
