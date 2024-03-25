def problem_p03923(input_data):
    n, a = list(map(int, input_data.split()))

    ans = 10**12 + 5

    for eat_num in range(45):

        if 2 ** (eat_num - 1) > n:
            break

        time = eat_num * a

        t = eat_num + 1

        l = 0
        r = n

        while r - l > 1:

            m = (r + l) / 2

            if m**t >= n:
                r = m

            else:
                l = m

        for j in range(t + 1):

            if r ** (t - j) * (r - 1) ** j < n:
                break

            surplus = j

        ans = min(ans, time + r * (t - surplus) + (r - 1) * surplus)

    return ans
