def problem_p03591():
    def slove():

        import sys

        input = sys.stdin.readline

        s = str(input().rstrip("\n"))

        print(("Yes" if s[:4] == "YAKI" else "No"))

    if __name__ == "__main__":

        slove()


problem_p03591()
