def problem_p03853():
    h, w = list(map(int, input().split()))

    c = []

    for _ in range(h):

        s = eval(input())

        c.append(s)

        c.append(s)

    print(("\n".join(c)))


problem_p03853()
