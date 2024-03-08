def problem_p03719():
    A, B, C = [int(x) for x in input().strip().split(" ")]

    if not C < A and not C > B:

        print("Yes")

    else:

        print("No")


problem_p03719()
