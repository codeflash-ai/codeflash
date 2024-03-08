def problem_p02379():
    import math

    x1, y1, x2, y2 = list(map(float, input().split(" ")))

    print(("{:.5f}".format(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))))


problem_p02379()
