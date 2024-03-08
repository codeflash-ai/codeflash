def problem_p03014():
    class UnionFind:

        def __init__(self, n):

            self.v = [
                -1 for _ in range(n)
            ]  # 根(負): 連結頂点数 * (-1) / 子(正): 根の頂点番号(0-indexed)

        def find(self, x):  # xを含む木における根の頂点番号を返す

            if self.v[x] < 0:  # (負)は根

                return x

            else:  # 根の頂点番号

                self.v[x] = self.find(
                    self.v[x]
                )  # uniteでは, 旧根に属する頂点の根が旧根のままなので更新

                return self.v[x]

        def unite(self, x, y):  # 違う根に属していたらrankが低くなるように連結

            x = self.find(x)

            y = self.find(y)

            if x == y:

                return

            if (
                -self.v[x] < -self.v[y]
            ):  # size比較, 　(-1) * (連結頂点数 * (-1)), (正)同士の大小比較

                x, y = y, x  # 連結頂点数が少ない方をyにすると, findでの更新回数が減る？

            self.v[x] += self.v[y]  # 連結頂点数の和を取る, 連結頂点数 * (-1)

            self.v[y] = (
                x  # 連結頂点数が少ないy(引数yの根の頂点番号)の根をx(引数xの根の頂点番号)にする
            )

        def root(self, x):

            return self.v[x] < 0  # (負)は根

        def same(self, x, y):

            return self.find(x) == self.find(y)  # 同じ根に属するか

        def size(self, x):

            return -self.v[self.find(x)]  # 連結頂点数を返す

    import sys

    input = sys.stdin.readline

    H, W = list(map(int, input().split()))

    s = tuple(eval(input()) for _ in range(H))

    uf_r = UnionFind(H * W)

    for r in range(H - 1):

        for c in range(W):

            if s[r][c] == "." and s[r + 1][c] == ".":

                uf_r.unite(r * W + c, (r + 1) * W + c)

    uf_c = UnionFind(H * W)

    for c in range(W - 1):

        for r in range(H):

            if s[r][c] == "." and s[r][c + 1] == ".":

                uf_c.unite(r * W + c, r * W + (c + 1))

    ans = 0

    for r in range(H):

        for c in range(W):

            if s[r][c] == "#":
                continue

            t = uf_r.size(r * W + c) + uf_c.size(r * W + c) - 1

            ans = max(ans, t)

    print(ans)


problem_p03014()
