def problem_p02702(input_data):
    import numpy as np

    s = input_data[::-1]

    length = len(s)

    a = np.zeros(length, dtype=int)

    d = np.zeros(length, dtype=int)

    ans = np.zeros(2019, dtype=int)

    x = 10

    a[0] = int(s[0])

    d[0] = a[0]

    ans[d[0]] += 1

    for i in range(1, length):

        a[i] = int(s[i]) * x % 2019

        d[i] = (d[i - 1] + a[i]) % 2019

        ans[d[i]] += 1

        x = x * 10 % 2019

    ans_c = ans[np.nonzero(ans)]

    return int(sum([ans_c[i] * (ans_c[i] - 1) for i in range(ans_c.shape[0])]) / 2) + ans[0]
