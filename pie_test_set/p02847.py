def problem_p02847(input_data):
    def main():

        S = eval(input_data)

        lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

        ans = 6 - lst.index(S)

        if ans == 0:

            ans = 7

        return ans

    if __name__ == "__main__":

        main()
