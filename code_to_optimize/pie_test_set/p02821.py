def problem_p02821():
    # E

    from itertools import accumulate

    from bisect import bisect_left

    def main():

        n, m = list(map(int, input().split()))

        A = list(map(int, input().split()))

        A.sort()

        def count(k):

            cnt = 0

            for a in A:

                cnt += bisect_left(A, k - a)

            return cnt

        ok = 0

        ng = 10**10 + 1

        while ng - ok > 1:

            mid = (ok + ng) // 2

            if count(mid) < n**2 - m:

                ok = mid

            else:

                ng = mid

        border = ok

        k = n**2 - count(border)

        Acc = [0] + list(accumulate(A))

        s = Acc[-1]

        ans = 0

        for a in A:

            idx = bisect_left(A, border - a)

            ans += a * (n - idx) + (s - Acc[idx])

        ans -= border * (k - m)

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02821()
