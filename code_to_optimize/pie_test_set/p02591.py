def problem_p02591():
    import sys

    h, *ppp = list(map(int, sys.stdin.buffer.read().split()))

    n = 1 << (h - 1)

    ppp = [p + n - 1 for p in ppp]

    MOD = 10**9 + 7

    cumprods = [1] * (2 * n)  # 根から頂点vまでの経路の累積積

    cumprods_rev = [1] * n  # ..の逆数

    cumprods_through = [0] * n  # 木1の根から木2の根まで、木1の i 番目の葉を経由する経路の積

    for i in range(2, 2 * n):

        cumprods[i] = cumprods[i >> 1] * i % MOD

    for i in range(n):

        cumprods_rev[i] = pow(cumprods[i], MOD - 2, MOD)

        cumprods_through[i] = cumprods[i + n] * cumprods[ppp[i]] % MOD

    # 現在考慮中の木1側のLCAを起点として、木2側の各頂点に至るまでの累積積

    # 木1からたどり着かないものは0

    cumprods_from_tree1 = [0] * (2 * n)

    ans = 0

    for lca in range(1, n):

        digit = h - lca.bit_length()

        leftmost_leaf_in_left_subtree = lca << digit

        leftmost_leaf_in_right_subtree = ((lca << 1) + 1) << (digit - 1)

        rightmost_leaf_in_right_subtree = (lca + 1) << digit

        rev = cumprods_rev[lca >> 1]

        for leaf in range(leftmost_leaf_in_left_subtree, leftmost_leaf_in_right_subtree):

            v = ppp[leaf - n]

            cp = cumprods_through[leaf - n] * rev % MOD

            while v > 1:

                cumprods_from_tree1[v] += cp * cumprods_rev[v >> 1] % MOD

                v >>= 1

        rev = cumprods_rev[lca]

        for leaf in range(leftmost_leaf_in_right_subtree, rightmost_leaf_in_right_subtree):

            v = ppp[leaf - n]

            cp = cumprods_through[leaf - n] * rev % MOD

            while v > 1:

                ans = (ans + cumprods_from_tree1[v ^ 1] * cp * cumprods_rev[v >> 2]) % MOD

                v >>= 1

        for leaf in range(leftmost_leaf_in_left_subtree, leftmost_leaf_in_right_subtree):

            v = ppp[leaf - n]

            while cumprods_from_tree1[v] != 0:

                cumprods_from_tree1[v] = 0

                v >>= 1

    print(ans)


problem_p02591()
