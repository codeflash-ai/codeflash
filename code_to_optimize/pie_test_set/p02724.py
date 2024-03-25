def problem_p02724(input_data):
    x = int(eval(input_data))

    # a,b = map(int, input_data.split())

    # l = list(map(int, input_data.split()))

    ans = 0

    r5 = int(x / 500)

    x = x - 500 * r5

    r1 = int(x / 5)

    return r5 * 1000 + r1 * 5
