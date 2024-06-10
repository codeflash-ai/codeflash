def problem_p03549(input_data):
    n, m = [int(i) for i in input_data.split()]

    ans = (n - m) * 100

    ans = (ans + 1900 * m) * 2**m

    return ans
