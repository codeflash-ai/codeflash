def problem_p03919():
    h, w, *a = open(0).read().split()
    b = a.index("snuke")
    print(([chr(i) for i in range(65, 91)][b % int(w)] + str(b // int(w) + 1)))


problem_p03919()
