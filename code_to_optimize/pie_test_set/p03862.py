def problem_p03862():
    def main():

        from collections import deque

        N, L = list(map(int, input().split()))

        (*a,) = list(map(int, input().split()))

        def solve(a):

            dq = deque()

            s = 0

            ans = 0

            for x in a:

                if len(dq) > 1:

                    t = dq.popleft()

                    s -= t

                eat = max(0, s + x - L)

                if eat:

                    x -= eat

                    ans += eat

                s += x

                dq.append(x)

            return ans

        ans = min(solve(a), solve(reversed(a)))

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03862()
