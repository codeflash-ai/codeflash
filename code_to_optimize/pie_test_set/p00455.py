def problem_p00455():
    a, b = 3600, 60

    for i in [1] * 3:

        h, m, s, x, y, z = list(map(int, input().split()))

        s = (x - h) * a + (y - m) * b + (z - s)

        print(s // a, s % a // b, s % b)


problem_p00455()
