def problem_p03209():
    def func(lv, id):

        sz = 4 * 2**lv - 3

        num = 2 * 2**lv - 1

        mid = (sz + 1) / 2

        if lv == 0:

            return 1

        if id == 1:

            return 0

        if id == mid:

            return num / 2 + 1

        if id == sz:

            return num

        if 1 < id < mid:

            return func(lv - 1, id - 1)

        if mid < id < sz:

            return func(lv - 1, id - sz / 2 - 1) + num / 2 + 1

    print(func(*list(map(int, input().split()))))


problem_p03209()
