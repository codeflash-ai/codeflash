def problem_p00137():
    def main():

        d = int(input())

        a = []

        for _ in range(d):

            a.append(int(input()))

        ans = []

        for x in range(d):

            ansans = []

            hoge = a[x]

            for y in range(10):

                hoge = list("{0:08d}".format(hoge**2))

                s = ""

                for z in range(2, 6):

                    s += hoge[z]

                hoge = int(s)

                ansans.append(hoge)

            ans.append(ansans)

        for x in range(1, d + 1):

            print("Case", x, end="")

            print(":")

            for y in ans[x - 1]:

                print(y)

    if __name__ == "__main__":

        main()


problem_p00137()
