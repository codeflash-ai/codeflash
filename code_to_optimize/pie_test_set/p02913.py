def problem_p02913():
    BASE = 9973

    MOD = 2**61 - 1

    class RollingHash:

        # 文字列Sのローリングハッシュを構築

        # 計算量: O(N)

        def __init__(self, S):

            # acc[i]はS[0:i]のハッシュ値となる

            self.acc = [0]

            a = 0

            # 累積和を計算するイメージ

            for i, c in enumerate(S):

                h = ord(c) - ord("a") + 1

                offset = pow(BASE, i, MOD)

                a = (a + (h * offset)) % MOD

                self.acc.append(a)

        # S[i:j]のハッシュ値を返す

        # 計算量: O(1)

        def hash(self, i, j):

            # 累積和を用いて部分区間の和を計算するイメージ

            # 注意: 基数によるオフセットをキャンセルする必要がある

            offset = pow(BASE, i, MOD)

            # 素数MODを法とする剰余類における除算は逆元を乗算することで計算できる

            # 素数MODを法とする剰余類におけるxの逆元はx^(MOD - 2)に等しい

            offset_inv = pow(offset, MOD - 2, MOD)

            return ((self.acc[j] - self.acc[i]) * offset_inv) % MOD

    N = int(eval(input()))

    S = eval(input())

    # Sのローリングハッシュを計算しておく

    rolling = RollingHash(S)

    # 長さmのダジャレがSに含まれるならば、長さm-1のダジャレも含まれる

    # つまり、長さmのダジャレがSに含まれるならば真となる述語をP(m)とすると、以下を満たす整数cが存在する:

    #   - 0以上c以下の整数mについてP(m)は真

    #   - cより大きい整数mについてP(m)は偽

    # ※ Pは単調関数ということ

    # このcを二分探索で発見すれば高速化できる！

    # P(ok)は真

    ok = 0

    # P(ng)は偽

    ng = N

    while ng - ok > 1:

        # 区間[ok, ng)の中央値をmとする

        m = (ok + ng) // 2

        # 長さmのダジャレが存在するか判定する

        # 長さmのformerのハッシュ値のキャッシュ

        former_cache = set()

        p = False

        # latterに着目してループを回す

        for i in range(m, N):

            # ダジャレ長を決め打ちしているのでjを直接計算できる

            j = i + m

            # S[i:j]がSをはみ出したらループ終了

            if j > N:

                break

            # formerのハッシュ値をキャッシュする

            former_cache.add(rolling.hash(i - m, i))

            # キャッシャに同一のハッシュ値が存在する場合（ハッシュ値が衝突した場合）、ダジャレになっていると判定できる

            if rolling.hash(i, j) in former_cache:

                p = True

            # ハッシュ値の計算、キャッシュ存在判定は共にO(1)で処理できる！

        # 区間[ok, ng) を更新

        if p:

            ok = m

        else:

            ng = m

    # whileループを抜けると区間[ok, ng)は[c, c + 1) になっている

    ans = ok

    print(ans)


problem_p02913()
