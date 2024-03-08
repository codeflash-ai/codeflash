def problem_p01522():
    n, k = list(map(int, input().split()))

    boats = [list(map(int, input().split())) for i in range(k)]

    r = eval(input())

    hate = [list(map(int, input().split())) for i in range(r)]

    blue = [0] * 51

    for i, j in hate:

        for boat in boats:

            if i in boat[1:] and j in boat[1:]:

                blue[i] = 1

                blue[j] = 1

    print(sum(blue))


problem_p01522()
