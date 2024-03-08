def problem_p03474():
    def slove():

        import sys

        input = sys.stdin.readline

        a, b = list(map(int, input().rstrip("\n").split()))

        s = str(input().rstrip("\n"))

        ls = list("1234567890")

        for i in range(len(s)):

            if i < a:

                if s[i] not in ls:

                    print("No")

                    exit()

            elif i == a:

                if s[i] != "-":

                    print("No")

                    exit()

            elif b <= i:

                if s[i] not in ls:

                    print("No")

                    exit()

        print("Yes")

    if __name__ == "__main__":

        slove()


problem_p03474()
