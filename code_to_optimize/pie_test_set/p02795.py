def problem_p02795():
    H = int(eval(input()))

    W = int(eval(input()))

    N = int(eval(input()))

    HW = H * W

    ans = H + W

    for h in range(0, H + 1):

        for w in range(0, W + 1):

            cntB = HW - (H - h) * (W - w)

            if cntB >= N:

                ans = min(ans, h + w)

    print(ans)


problem_p02795()
