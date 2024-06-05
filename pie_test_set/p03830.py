def problem_p03830(input_data):
    MOD = 10**9 + 7

    n = int(eval(input_data))

    res = 1

    ex = [0 for _ in range(n + 1)]

    for i in range(1, n + 1):

        for j in range(2, i + 1):

            if i % j == 0:

                while i % j == 0:

                    ex[j] += 1

                    i //= j

    ans = 1

    for i in range(n + 1):

        ans *= ex[i] + 1

        ans %= MOD

    return ans
