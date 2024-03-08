def problem_p00106():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0106&lang=jp

    """

    import sys

    def solve(amount):
        """

        :param amount: ?????\???????????°?????????

        :return: ?????????

        """

        best_price = (amount + 199) // 200 * 380

        for a in range(amount // 200 + 1):

            for b in range(amount // 300 + 1):

                for c in range(amount // 500 + 1):

                    for a1 in range(amount // 1000 + 1):

                        for b1 in range(amount // 1200 + 1):

                            for c1 in range(amount // 1500 + 1):

                                if (
                                    a * 200 + b * 300 + c * 500 + a1 * 1000 + b1 * 1200 + c1 * 1500
                                    == amount
                                ):

                                    price = (
                                        a * 380
                                        + b * 550
                                        + c * 850
                                        + a1 * 1520
                                        + b1 * 1870
                                        + c1 * 2244
                                    )

                                    if price < best_price:

                                        best_price = price

        return best_price

    def main(args):

        while True:

            amount = int(eval(input()))

            if amount == 0:

                break

            result = solve(amount)

            print(result)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00106()
