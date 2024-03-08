def problem_p00468():
    for e in iter(input, "0"):

        R = [set() for _ in [0] * -~int(e)]

        for _ in [0] * int(eval(input())):

            a, b = list(map(int, input().split()))

            R[a] |= {b}

            if a - 1:
                R[b] |= {a}

        for m in set(R[1]):

            R[1] |= R[m]

        print((len(R[1])))


problem_p00468()
