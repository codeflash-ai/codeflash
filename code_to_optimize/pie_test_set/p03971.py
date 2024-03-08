def problem_p03971():
    N, A, B = list(map(int, input().split()))

    S = eval(input())

    pa = pb = 0

    for c in S:

        if c == "a" and pa + pb < A + B:

            print("Yes")

            pa += 1

        elif c == "b" and pa + pb < A + B and pb < B:

            print("Yes")

            pb += 1

        else:

            print("No")


problem_p03971()
