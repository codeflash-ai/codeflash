def problem_p02921():
    import itertools

    import fractions

    def main():

        s = eval(input())

        t = eval(input())

        cnt = 0

        for i in range(3):

            if s[i] == t[i]:

                cnt += 1

        print(cnt)

    if __name__ == "__main__":

        main()


problem_p02921()
