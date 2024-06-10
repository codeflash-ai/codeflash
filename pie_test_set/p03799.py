def problem_p03799(input_data):
    s, c = list(map(int, input_data.split()))

    ans = min(s, c // 2)

    c -= ans * 2

    ans += c // 4

    return ans
