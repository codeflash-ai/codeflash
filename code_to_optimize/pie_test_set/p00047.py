def problem_p00047():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0047

    Cup Game

    """

    import sys

    def main(args):

        ball_position = "A"

        for line in sys.stdin:

            exchanged_cups = line.strip().split(",")

            if ball_position in exchanged_cups:

                exchanged_cups.remove(ball_position)

                ball_position = exchanged_cups[0]

        print(ball_position)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00047()
