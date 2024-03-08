def problem_p03185():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    from collections import deque

    N, C, *H = list(map(int, read().split()))

    class CHT:
        """

        f_i = a_ix + b_i とする。f_i の追加および、min_i f(x) の取得ができるデータ構造。

        ただし、傾き a_i は昇順に追加されなければならない。

        また、クエリ x も昇順に実行されなければならない。

        """

        def __init__(self):

            self.funcs = deque()

        def add(self, a, b):

            funcs = self.funcs

            while len(funcs) >= 2:

                a1, b1 = funcs[-2]

                a2, b2 = funcs[-1]

                if (a2 - a1) * (b - b2) < (b2 - b1) * (a - a2):

                    break

                funcs.pop()

            funcs.append((a, b))

        def query(self, x):

            funcs = self.funcs

            a, b = funcs[0]

            y = a * x + b

            while len(funcs) >= 2:

                a2, b2 = self.funcs[1]

                y2 = a2 * x + b2

                if y < y2:

                    break

                y = y2

                self.funcs.popleft()

            return y

    dp = [0] * N

    cht = CHT()

    h = H[0]

    cht.add(-2 * h, h * h)

    for i, h in enumerate(H[1:], 1):

        x = cht.query(h) + h * h + C

        dp[i] = x

        cht.add(-2 * h, h * h + x)

    answer = dp[-1]

    print(answer)


problem_p03185()
