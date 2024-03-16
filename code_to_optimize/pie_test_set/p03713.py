def problem_p03713(input_data):
    H, W = list(map(int, input_data.split()))

    ans = float("inf")

    for h in range(1, H):

        S = [h * W]

        w = W // 2

        S += [(H - h) * w]

        S += [(H - h) * (W - w)]

        ans = min(ans, max(S) - min(S))

    for h in range(1, H):

        S = [h * W]

        hb = (H - h) // 2

        S += [hb * W]

        S += [(H - h - hb) * W]

        ans = min(ans, max(S) - min(S))

    for w in range(1, W):

        S = [H * w]

        h = H // 2

        S += [h * (W - w)]

        S += [(H - h) * (W - w)]

        ans = min(ans, max(S) - min(S))

    for w in range(1, W):

        S = [w * H]

        wb = (W - w) // 2

        S += [wb * H]

        S += [(W - w - wb) * H]

        ans = min(ans, max(S) - min(S))

    return ans
