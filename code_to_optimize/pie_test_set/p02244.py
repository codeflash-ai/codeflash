def problem_p02244():
    from copy import deepcopy, copy

    class QueenMAP:

        __slots__ = ["yoko", "tate", "naname1", "naname2", "MAP"]

        def __init__(self):

            self.yoko = set()

            self.tate = set()

            self.naname1 = set()

            self.naname2 = set()

            self.MAP = [["."] * 8 for _ in range(8)]

        def add(self, y, x):

            self.MAP[y][x] = "Q"

            self.yoko.add(y)

            self.tate.add(x)

            self.naname1.add(y - x)

            self.naname2.add(x + y)

        def check(self, y, x):

            if x in self.tate or (y - x) in self.naname1 or (x + y) in self.naname2:

                return False

            return True

        def allcheck(self):

            for i in range(8):

                if not "Q" in self.MAP[i]:

                    return False

            return True

    def MAIN():

        f = lambda M: "\n".join("".join(map(str, m)) for m in M)

        QM = QueenMAP()

        n = int(eval(input()))

        for _ in range(n):

            a, b = list(map(int, input().split()))

            QM.add(a, b)

        dp = [(deepcopy(QM), n)]

        while dp:

            Q, cnt = dp.pop()

            if cnt == 8:

                if Q.allcheck():

                    print((f(Q.MAP)))

                    break

                continue

            cnt += 1

            for i in range(8):

                if i in Q.yoko:

                    continue

                for j in range(8):

                    if Q.check(i, j):

                        CQ = deepcopy(Q)

                        CQ.add(i, j)

                        dp.append((CQ, cnt))

    MAIN()


problem_p02244()
