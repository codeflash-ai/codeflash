def problem_p02861():
    import math

    import itertools

    N = int(eval(input()))

    lst = []

    for i in range(N):

        lst.append(list(map(int, input().split())))

    routes = list(itertools.permutations(lst))

    def distance(a, b):

        return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)) ** 0.5

    routes = [i for i in routes]

    distancies = []

    for i, r in enumerate(routes):

        for j, a in enumerate(r):

            if j >= len(r) - 1:
                break

            b = r[j + 1]

            ab_dist = distance(a, b)

            distancies.append(ab_dist)

    print((sum(distancies) / len(routes)))


problem_p02861()
