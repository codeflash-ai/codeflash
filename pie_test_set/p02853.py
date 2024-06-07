def problem_p02853(input_data):
    x, y = list(map(int, input_data.split()))

    amount = 0

    if x < 4:

        amount += (4 - x) * 100000

    if y < 4:

        amount += (4 - y) * 100000

    if x == 1 and y == 1:

        amount += 400000

    return amount
