def problem_p02554(input_data):
    N = int(eval(input_data))

    ans = 10**N - 2 * 9**N + 8**N

    return ans % (10**9 + 7)
