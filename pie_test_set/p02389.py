def problem_p02389(input_data):
    [a, b] = list(map(int, input().split()))

    return "%s %s" % (a * b, 2 * (a + b))
