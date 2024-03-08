def problem_p03229():
    import sys

    import collections

    sys.setrecursionlimit(10**8)

    input = sys.stdin.readline

    def main():

        N = int(eval(input()))

        A = [int(eval(input())) for _ in range(N)]

        if N == 2:

            print((abs(A[0] - A[1])))

            return

        A.sort()

        q = collections.deque(A)

        ans = 0

        i = 0

        c = [q.popleft(), q.pop()]

        ans += abs(c[0] - c[1])

        while q:

            if len(q) >= 2:

                a = q.popleft()

                b = q.pop()

                al = abs(a - c[0])

                ar = abs(a - c[1])

                bl = abs(b - c[0])

                br = abs(b - c[1])

                ma = max([al, ar, bl, br])

                ans += ma

                if ma == al:

                    c[0] = a

                    q.append(b)

                elif ma == ar:

                    c[1] = a

                    q.append(b)

                elif ma == bl:

                    c[0] = b

                    q.appendleft(a)

                else:

                    c[1] = b

                    q.appendleft(a)

            else:

                a = q.pop()

                ans += max(abs(c[0] - a), abs(c[1] - a))

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03229()
