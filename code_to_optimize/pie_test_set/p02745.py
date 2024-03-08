def problem_p02745():
    a = eval(input())

    b = eval(input())

    c = eval(input())

    # 文字のマッチング(?はワイルドカード)

    def match(a, b):

        return a == "?" or b == "?" or a == b

    ab = [True] * 100000  # Aに対して(i-50000)個左にスライドした位置にBを置けるか？

    ac = [True] * 100000  # Aに対して(i-50000)個左にスライドした位置にCを置けるか？

    bc = [True] * 100000  # Bに対して(i-50000)個左にスライドした位置にCを置けるか？

    def check(a, b, ab):

        for i in range(len(a)):

            for j in range(len(b)):

                if not match(a[i], b[j]):

                    ab[i - j + 50000] = False

    check(a, b, ab)

    check(a, c, ac)

    check(b, c, bc)

    len_max = 2000

    len_a = len(a)

    len_b = len(b)

    len_c = len(c)

    ans = len_a + len_b + len_c

    # i = BをAに対していくつスライドして置くか

    for i in range(-2 * len_max, 2 * len_max + 1):

        # j = CをAに対していくつスライドして置くか

        for j in range(-2 * len_max, 2 * len_max + 1):

            if ab[i + 50000] and ac[j + 50000] and bc[j - i + 50000]:

                ans = min(ans, max(len_a, len_b + i, len_c + j) - min(0, i, j))

    print(ans)


problem_p02745()
