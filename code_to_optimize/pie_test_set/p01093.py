def problem_p01093():
    while int(eval(input())) > 0:

        s = sorted(map(int, input().split()))

        print((min(abs(a - b) for (a, b) in zip(s, s[1:]))))


problem_p01093()
