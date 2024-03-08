def problem_p02771():
    import numpy

    A, B, C = input().split()

    a = A

    b = B

    c = C

    if a == b and a == c:

        print("No")

    elif a == b or b == c or a == c:

        print("Yes")

    elif a != b or b != c:

        print("No")


problem_p02771()
