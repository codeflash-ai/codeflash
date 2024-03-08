def problem_p03827():
    from collections import defaultdict

    def main():

        N = int(eval(input()))

        S = str(eval(input()))

        x = 0

        ans = 0

        for c in S:

            x += {"I": 1, "D": -1}[c]

            ans = max(ans, x)

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03827()
