def problem_p02904():
    class SWAG:

        __slots__ = ("unit", "f", "fold_r", "fold_l", "data_r", "data_l")

        def __init__(self, unit, f, data):

            self.unit = unit

            self.f = f

            self.fold_r = [unit]

            self.fold_l = [unit]

            self.data_r = []

            self.data_l = []

            sep = len(data) // 2

            for t in data[sep:]:

                self.append(t)

            for t in data[sep - 1 :: -1]:

                self.appendleft(t)

        def append(self, x):

            self.fold_r.append(self.f(self.fold_r[-1], x))

            self.data_r.append(x)

        def appendleft(self, x):

            self.fold_l.append(self.f(self.fold_l[-1], x))

            self.data_l.append(x)

        def pop(self):

            if not self.data_r:

                self.__init__(self.unit, self.f, self.data_l[::-1])

            self.fold_r.pop()

            return self.data_r.pop()

        def popleft(self):

            if not self.data_l:

                self.__init__(self.unit, self.f, self.data_r)

            self.fold_l.pop()

            return self.data_l.pop()

        def fold_all(self):

            return self.f(self.fold_r[-1], self.fold_l[-1])

    n, k, *p = list(map(int, open(0).read().split()))

    c = [0]

    for a, b in zip(p, p[1:]):
        c += (c[-1] + (a < b),)

    *c, f = [b - a == k - 1 for a, b in zip(c, c[k - 1 :])]

    x = not f

    s_min = SWAG(10**18, min, p[: k - 1])

    s_max = SWAG(0, max, p[1:k])

    for i, (a, b, c) in enumerate(zip(p, p[k:], c)):

        f |= c

        s_min.append(p[i + k - 1])

        s_max.append(b)

        if not c and (a != s_min.fold_all() or b != s_max.fold_all()):
            x += 1

        s_min.popleft()

        s_max.popleft()

    print((x + f))


problem_p02904()
