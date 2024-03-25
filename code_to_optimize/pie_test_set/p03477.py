def problem_p03477(input_data):
    def main():

        A, B, C, D = list(map(int, input_data.split()))

        hoge = A + B

        fuga = C + D

        if hoge > fuga:

            return "Left"

        elif hoge == fuga:

            return "Balanced"

        else:

            return "Right"

    if __name__ == "__main__":

        main()
