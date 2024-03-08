def problem_p02384():
    # coding: utf-8

    # ?????????????????¨????????????

    class Dice(object):

        def __init__(self):

            # ???????????????????????°

            # ????????¶???

            self.dice = (2, 5), (3, 4), (1, 6)  # x, y, z

            self.ax = [[0, False], [1, False], [2, False]]

            self.axmap = [0, 1, 2]

            self.mm = {"N": (0, 2), "S": (2, 0), "E": (1, 2), "W": (2, 1), "R": (0, 1), "L": (1, 0)}

        def rotate(self, dir):

            def rot(k, r):

                # k?????????????????????????????????????????§?§????

                # r?????????????????¢???????§????

                t = self.axmap[r]

                self.axmap[k], self.axmap[r] = t, self.axmap[k]

                self.ax[t][1] = not self.ax[t][1]

            rot(*self.mm[dir])

        def top(self):

            z = self.ax[self.axmap[2]]

            return self.dice[z[0]][z[1]]

        def right(self):

            y = self.ax[self.axmap[1]]

            return self.dice[y[0]][y[1]]

        def front(self):

            x = self.ax[self.axmap[0]]

            return self.dice[x[0]][x[1]]

    if __name__ == "__main__":

        dice = Dice()

        labels = input().split()

        q = int(eval(input()))

        for _ in range(q):

            a, b = input().split()

            p = labels.index(a) + 1

            for _ in range(4):

                if p == dice.top():

                    break

                dice.rotate("N")

            for _ in range(4):

                if p == dice.top():

                    break

                dice.rotate("E")

            p = labels.index(b) + 1

            for _ in range(4):

                if p == dice.front():

                    break

                dice.rotate("R")

            print((labels[dice.right() - 1]))


problem_p02384()
