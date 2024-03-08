def problem_p02256():
    # http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ALDS1_1_B&lang=jp

    # ??????????????????????????????

    # ?????§??¬?´???°???????????????????????????????????¨????????¨?¨????????????????

    def gcd(x, y):

        tmp = y

        if y > x:

            y = x

            x = tmp

        while not y == 0:

            y = x % y

            x = tmp

            tmp = y

        return x

    def main():

        target = [int(a) for a in input().split()]

        print((gcd(target[0], target[1])))

    if __name__ == "__main__":

        main()


problem_p02256()
