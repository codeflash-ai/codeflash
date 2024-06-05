def problem_p03945(input_data):
    s = eval(input_data)

    len_s = len(s)

    cnt = [0, 0]

    for i in range(1, len(s)):

        memo_0 = s[i - 1]

        memo_1 = s[len_s - i]

        if s[i] != memo_0:

            cnt[0] += 1

        if s[len_s - i - 1] != memo_1:

            cnt[1] += 1

    return min(cnt)
