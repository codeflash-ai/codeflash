def problem_p02659():
    a, b = input().split()
    print((int(a) * int(b[:-3] + b[-2:]) // 100))


problem_p02659()
