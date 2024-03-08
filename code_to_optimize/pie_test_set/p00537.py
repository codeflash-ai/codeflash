def problem_p00537():
    class Bit:

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

    n, m = list(map(int, input().split()))

    bit = Bit(n + 1)

    cities = list(map(int, input().split()))

    prev = next(cities)

    for cur in cities:

        if prev < cur:

            bit.add(prev, 1)

            bit.add(cur, -1)

        else:

            bit.add(cur, 1)

            bit.add(prev, -1)

        prev = cur

    ans = 0

    for i in range(1, n):

        a, b, c = list(map(int, input().split()))

        bep = c // (a - b)

        cnt = bit.sum(i)

        if cnt > bep:

            ans += b * cnt + c

        else:

            ans += a * cnt

    print(ans)


problem_p00537()
