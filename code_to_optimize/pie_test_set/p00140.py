def problem_p00140():
    bus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(eval(input())):

        s, t = list(map(int, input().split()))

        if s > 5:

            print(" ".join(map(str, bus[s : bus.index(t, s) + 1])))

        else:

            if s < t:

                print(" ".join(map(str, range(s, t + 1))))

            else:

                print(" ".join(map(str, range(s, t - 1, -1))))


problem_p00140()
