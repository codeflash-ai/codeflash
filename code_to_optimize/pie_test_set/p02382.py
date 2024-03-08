def problem_p02382():
    import math

    def manhattan(xs, ys):

        return sum([abs(x - y) for x, y in zip(xs, ys)])

    def euclid(xs, ys):

        return math.sqrt(sum([(x - y) ** 2 for x, y in zip(xs, ys)]))

    def l3(xs, ys):

        return sum([abs(x - y) ** 3 for x, y in zip(xs, ys)]) ** (1 / 3)

    def chev(xs, ys):

        return max([abs(x - y) for x, y in zip(xs, ys)])

    def norm():

        n = int(eval(input()))

        xs = list(map(int, input().split()))

        ys = list(map(int, input().split()))

        print((manhattan(xs, ys)))

        print((euclid(xs, ys)))

        print((l3(xs, ys)))

        print((chev(xs, ys)))

    norm()


problem_p02382()
