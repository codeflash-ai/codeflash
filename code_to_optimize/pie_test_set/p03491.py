def problem_p03491():
    import sys

    from collections import deque

    sys.setrecursionlimit(10000)

    INF = float("inf")

    N, L = list(map(int, input().split()))

    S = set()

    for _ in range(N):

        S.add(eval(input()))

    # 追加できるのが 01... だけで

    # L が 4 とすると

    # 01: 次はない

    # 010: 011 がある

    # 011: 010 がある

    # 0100: 011, 0101 がある

    # 0101: 011, 0100

    # 0110: 010, 0111

    # 0111: 010, 0110

    # 候補の長さと L だけによって次の候補の長さと数が決まる

    def grundy(size):
        """

        :param int size: 候補の長さ

        :return:

        """

        # gs = {0}

        # g = 0

        # for sz in range(size + 1, L + 1):

        #     g ^= grundy(sz)

        #     gs.add(g)

        #

        # i = 0

        # while i in gs:

        #     i += 1

        # return i

        i = 1

        while (L - size + 1) % i == 0:

            i *= 2

        return i // 2

    # S の各要素からトライ木つくる

    trie = {}

    for s in S:

        t = trie

        for c in s:

            if c not in t:

                t[c] = {}

            t = t[c]

    # 候補文字列の長さのリスト

    ok_size_list = []

    children = deque()

    children.append((trie, 0))

    while len(children) > 0:

        node, size = children.popleft()

        # 0 か 1 か片方しか行けなかったら、行けない方は S の prefix にならない

        # 両方行けない場合はどう追加しても S のいずれかが prefix になってしまうからダメ

        if len(node) == 1:

            ok_size_list.append(size + 1)

        for c, child in list(node.items()):

            children.append((child, size + 1))

    g = 0

    for size in ok_size_list:

        g ^= grundy(size)

    if g != 0:

        print("Alice")

    else:

        print("Bob")


problem_p03491()
