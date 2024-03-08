def problem_p03836():
    sx, sy, tx, ty = list(map(int, input().split()))

    dx = tx - sx

    dy = ty - sy

    direction1 = "U" * dy + "R" * dx

    direction2 = "D" * dy + "L" * dx

    direction3 = "L" + "U" * (dy + 1) + "R" * (dx + 1) + "D"

    direction4 = "R" + "D" * (dy + 1) + "L" * (dx + 1) + "U"

    print((direction1 + direction2 + direction3 + direction4))


problem_p03836()
