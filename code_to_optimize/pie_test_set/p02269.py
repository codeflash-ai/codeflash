def problem_p02269():
    import sys

    n = int(eval(input()))

    htable = [0 for _ in range(16777216)]

    def to_int(s):

        if s == "A":

            return 1

        elif s == "C":

            return 2

        elif s == "G":

            return 3

        else:

            return 4

    def to_hash(string):

        h = 0

        for i, s in enumerate(string):

            ints = to_int(s)

            h += 4**i * ints

        return h

    def insert(string):

        h = to_hash(string)

        htable[h] = 1

    def find(string):

        h = to_hash(string)

        if htable[h]:

            print("yes")

        else:

            print("no")

    for line in sys.stdin:

        cmd, string = line.strip().split()

        if cmd == "insert":

            insert(string)

        else:

            find(string)


problem_p02269()
