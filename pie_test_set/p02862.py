def problem_p02862(input_data):
    X, Y = list(map(int, input_data.split()))

    import sys

    if (2 * Y - X) % 3 != 0 or (2 * X - Y) % 3 != 0:

        return 0

        sys.exit()

    if (2 * Y - X) < 0 or (2 * X - Y) < 0:

        return 0

        sys.exit()

    x = (2 * Y - X) // 3

    y = (2 * X - Y) // 3

    # (x+y)Cxを求める

    fac = [0 for i in range(x + y + 1)]

    inv = [0 for i in range(x + y + 1)]

    finv = [0 for i in range(x + y + 1)]

    # 初期条件

    p = 1000000007

    fac[0] = fac[1] = 1

    inv[1] = 1

    finv[0] = finv[1] = 1

    # テーブルの作成

    for i in range(2, x + y + 1):

        fac[i] = fac[i - 1] * i % p

        # p=(p//a)*a+(p%a) a^(-1)=-(p//a)*(p%a)^(-1)

        inv[i] = (-(p // i) * inv[p % i]) % p

        finv[i] = finv[i - 1] * inv[i] % p

    # 求める

    return (fac[x + y] * finv[x] % p) * finv[y] % p
