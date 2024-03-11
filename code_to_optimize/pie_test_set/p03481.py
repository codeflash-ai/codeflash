def problem_p03481(input_data):
    import sys

    input = sys.stdin.readline

    x, y = [int(x) for x in input_data.split()]

    ans = 0

    while x <= y:

        x *= 2

        ans += 1

    return ans
