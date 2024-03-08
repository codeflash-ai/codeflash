def problem_p04030():
    def main():

        s = eval(input())

        res = ""

        for i in s:

            if i == "B":

                res = res[:-1]

            else:

                res += i

        print(res)

    if __name__ == "__main__":

        main()


problem_p04030()
