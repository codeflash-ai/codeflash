def problem_p03679():
    x, a, b = list(map(int, input().split()))

    if b - a <= x and b - a <= 0:

        print("delicious")

    elif b - a <= x:

        print("safe")

    else:

        print("dangerous")


problem_p03679()
