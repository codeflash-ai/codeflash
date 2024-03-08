def problem_p03623():
    x, a, b = list(map(int, input().split()))

    A = abs(x - a)

    B = abs(x - b)

    if A < B:

        print("A")

    else:

        print("B")


problem_p03623()
