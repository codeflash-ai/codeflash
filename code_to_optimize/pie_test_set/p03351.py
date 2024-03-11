def problem_p03351(input_data):
    def main():

        # 1.input

        a, b, c, d = list(map(int, input_data.split()))

        # 2. judge

        if abs(a - c) <= d:

            return "Yes"

        elif abs(a - b) <= d and abs(b - c) <= d:

            return "Yes"

        else:

            return "No"

    if __name__ == "__main__":

        main()
