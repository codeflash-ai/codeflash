def problem_p00631():
    while 1:

        n = eval(input())

        if n == 0:
            break

        a = list(map(int, input().split()))

        s = sum(a)

        ref = [0]

        for i in a:

            ref += [i + j for j in ref if j < s / 2]

        print(min(abs(s - 2 * i) for i in ref))


problem_p00631()
