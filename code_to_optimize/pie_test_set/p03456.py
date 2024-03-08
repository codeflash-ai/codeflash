def problem_p03456():
    import math

    a, b = input().split()

    S = a + b

    K = int(S)

    if math.sqrt(K) % 1 == 0:

        print("Yes")

    else:

        print("No")


problem_p03456()
