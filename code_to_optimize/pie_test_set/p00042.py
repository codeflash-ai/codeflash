def problem_p00042():
    c = 0

    for W in iter(input, "0"):

        c += 1

        W = int(W)

        d = [0] * -~W

        for _ in [0] * int(eval(input())):

            v, w = list(map(int, input().split(",")))

            for i in range(W, w - 1, -1):

                if d[i] < d[i - w] + v:
                    d[i] = d[i - w] + v

        print(f"Case {c}:\n{d[W]}\n{d.index(d[W])}")


problem_p00042()
