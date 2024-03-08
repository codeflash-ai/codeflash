def problem_p03857():
    # Union Find

    # 木の根を求める

    def find(x, par):

        if par[x] == x:

            return x

        else:

            return find(par[x], par)

    # xとyの属する集合の併合

    def unite(x, y, par, rank):

        x = find(x, par)

        y = find(y, par)

        if x != y:

            # xとyの属している集合が異なる時

            if rank[x] < rank[y]:

                par[x] = y

            else:

                par[y] = x

                if rank[x] == rank[y]:

                    rank[x] += 1

    # xとyが同じ集合に属するかの判定

    def same(x, y, par):

        return find(x, par) == find(y, par)

    n, k, L = list(map(int, input().split()))

    rpar = []  # 親

    tpar = []

    rrank = [0] * n  # 木の深さ

    trank = [0] * n

    # 初期化

    for i in range(n):

        # par[i]:i rank[i]:0

        rpar.append(i)

        tpar.append(i)

    for i in range(k):

        x, y = list(map(int, input().split()))

        unite(x - 1, y - 1, rpar, rrank)

    for i in range(L):

        x, y = list(map(int, input().split()))

        unite(x - 1, y - 1, tpar, trank)

    num = []

    d = {}

    for i in range(n):

        num.append(find(i, rpar) + find(i, tpar) * 10**6)

        if num[-1] in d:

            d[num[-1]] += 1

        else:

            d[num[-1]] = 1

    res = ""

    for i in range(n):

        if i != 0:

            res += " "

        res += str(d[num[i]])

    print(res)


problem_p03857()
