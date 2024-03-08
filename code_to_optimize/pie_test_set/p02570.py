def problem_p02570():
    D, T, S = list(map(int, input().split()))

    a = T * S - D

    if a >= 0:

        print("Yes")

    else:

        print("No")


problem_p02570()
