def problem_p01852():
    n = int(eval(input()))

    print((0 if n == 0 else len(str(bin(n))[2:])))


problem_p01852()
