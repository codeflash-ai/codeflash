def problem_p03039(input_data):
    H, W, K = list(map(int, input_data.split()))

    MOD = 10**9 + 7

    Hd = sum([d * (H - d) * (W**2) for d in range(H)])

    Wd = sum([d * (W - d) * (H**2) for d in range(W)])

    # 階乗 & 逆元計算

    factorial = [1]

    inverse = [1]

    for i in range(1, H * W + 2):

        factorial.append(factorial[-1] * i % MOD)

        inverse.append(pow(factorial[-1], MOD - 2, MOD))

    ans = (Hd + Wd) * factorial[H * W - 2] * inverse[K - 2] * inverse[H * W - K] % MOD

    return ans
