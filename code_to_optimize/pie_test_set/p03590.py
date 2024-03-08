def problem_p03590():
    n, k = list(map(int, input().split()))

    ab = [list(map(int, input().split())) for _ in range(n)]

    ans = sum(b for a, b in ab if a | k == k)

    k_bin = bin(k)

    # print(k_bin)

    for i in range(len(k_bin) - 2):

        if (k >> i) & 1:

            x = k_bin[: -(i + 1)] + "0" + "1" * i

            x = int(x, 0)

            cand = 0

            for a, b in ab:

                if a | x == x:

                    cand += b

            # print(i, bin(x), cand)

            ans = max(ans, cand)

    print(ans)


problem_p03590()
