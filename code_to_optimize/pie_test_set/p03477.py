def problem_p03477():
    def main():

        A, B, C, D = list(map(int, input().split()))

        hoge = A + B

        fuga = C + D

        if hoge > fuga:

            print("Left")

        elif hoge == fuga:

            print("Balanced")

        else:

            print("Right")

    if __name__ == "__main__":

        main()


problem_p03477()
