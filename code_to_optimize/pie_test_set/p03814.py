def problem_p03814():
    # Problem https://atcoder.jp/contests/abc053/tasks/abc053_b

    # Python 1st Try

    import copy

    class Problem:

        def __init__(self, stringdata):

            self.stringdata = stringdata

        def solver(self):

            strcont = copy.copy(self.stringdata)

            result = 0

            rightposition = 0

            leftposition = len(strcont)

            j = 0

            while True:

                if strcont[j] == "A":

                    rightposition = j

                    break

                else:

                    j = j + 1

            j = leftposition - 1

            while True:

                if strcont[j] == "Z":

                    leftposition = j

                    break

                else:

                    j = j - 1

            result = leftposition - rightposition + 1

            return result

    if __name__ == "__main__":

        s = eval(input())

        print(("{}".format(Problem(s).solver())))


problem_p03814()
