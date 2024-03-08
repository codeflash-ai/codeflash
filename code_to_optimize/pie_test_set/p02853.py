def problem_p02853():
    x, y = list(map(int, input().split()))

    amount = 0

    if x < 4:

        amount += (4 - x) * 100000

    if y < 4:

        amount += (4 - y) * 100000

    if x == 1 and y == 1:

        amount += 400000

    print(amount)


problem_p02853()
