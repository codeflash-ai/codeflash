def problem_p00436():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0513

    AC

    """

    import sys

    from sys import stdin

    from itertools import chain

    input = stdin.readline

    def flatten(listOfLists):

        "Flatten one level of nesting"

        return chain.from_iterable(listOfLists)

    def main(args):

        global cards

        n = int(eval(input()))

        m = int(eval(input()))

        cards = [x for x in range(1, (2 * n) + 1)]

        for _ in range(m):

            op = int(eval(input()))

            if op == 0:

                # shuffle()

                temp = [[y, b] for y, b in zip(cards[:n], cards[n:])]

                cards = list(flatten(temp))

            else:

                # cut(k)

                cards = cards[op:] + cards[:op]

        print(("\n".join(map(str, cards))))

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00436()
