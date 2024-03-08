def problem_p02394():
    a = input().split()

    b = list(map(int, a))

    W = b[0]

    H = b[1]

    x = b[2]

    y = b[3]

    r = b[4]

    if (r <= x <= (W - r)) and (r <= y <= (H - r)):

        print("Yes")

    else:

        print("No")


problem_p02394()
