def problem_p02788():
    import sys

    input = sys.stdin.buffer.readline

    def main():

        n, d, a = list(map(int, input().split()))

        XH = []

        for i in range(n):

            x, h = list(map(int, input().split()))

            XH.append((x, h))

        XH.sort()

        from collections import deque

        q = deque()

        ans = 0

        t = 0

        import copy

        for x, h in XH:

            while q:

                r, s = q.popleft()

                if r + d < x:

                    t -= s

                else:

                    q.appendleft((r, s))

                    break

            h -= t

            if h > 0:

                if h % a == 0:

                    b = h // a

                    ans += b

                    t += a * b

                    q.append((x + d, a * b))

                else:

                    b = h // a + 1

                    ans += b

                    t += a * b

                    q.append((x + d, a * b))

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02788()
