def problem_p03351():
    def main():

        # 1.input

        a, b, c, d = list(map(int, input().split()))

        # 2. judge

        if abs(a - c) <= d:

            print("Yes")

        elif abs(a - b) <= d and abs(b - c) <= d:

            print("Yes")

        else:

            print("No")

    if __name__ == "__main__":

        main()


problem_p03351()
