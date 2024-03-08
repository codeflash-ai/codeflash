def problem_p02421():
    n = int(eval(input()))

    tp = hp = 0

    for nc in range(n):

        (tc, hc) = input().split()

        if tc == hc:

            tp += 1

            hp += 1

        elif tc > hc:

            tp += 3

        elif tc < hc:

            hp += 3

    print((tp, hp))


problem_p02421()
