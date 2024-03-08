def problem_p02854():
    n = int(eval(input()))

    stick = list(map(int, input().split()))

    total = sum(stick)

    mid = total // 2

    cum = 0

    midi = 0

    for i, block in enumerate(stick):

        cum += block

        if cum >= mid:

            midi = i

            break

    l1 = sum(stick[:midi])

    r1 = sum(stick[midi:])

    diff1 = abs(l1 - r1)

    l2 = l1 + stick[midi]

    r2 = r1 - stick[midi]

    diff2 = abs(l2 - r2)

    print((min(diff1, diff2)))


problem_p02854()
