def problem_p01486(input_data):
    S = eval(input_data)

    L = len(S)

    dp = [[False] * (L + 1) for _ in range(L + 1)]

    # [left, right)

    for i in range(L + 1):

        dp[i][i] = True

    for w in range(3, L + 1):

        for left in range(L - w + 1):

            right = left + w

            if S[left] != "m" or S[right - 1] != "w":

                continue

            for mid in range(left + 1, right):

                if S[mid] == "e" and dp[left + 1][mid] and dp[mid + 1][right - 1]:

                    dp[left][right] = True

                    break

    return "Cat" if dp[0][L] else "Rabbit"
