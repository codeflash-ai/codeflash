def problem_p02996():

    n = int(eval(input()))

    abn = [list(map(int, input().split())) for _ in range(n)]

    abn.sort()

    abn.sort(key=lambda x: x[1])

    # print(abn)

    ts = 0  # time_stamp

    enable = True

    for abi in abn:

        a, b = abi

        if not ts + a <= b:

            enable = False

            break

        else:

            ts += a

    print(("Yes" if enable else "No"))


problem_p02996()
