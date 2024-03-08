def problem_p02847():
    def main():

        S = eval(input())

        lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

        ans = 6 - lst.index(S)

        if ans == 0:

            ans = 7

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02847()
