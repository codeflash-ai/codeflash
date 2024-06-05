def problem_p03591(input_data):
    def slove():

        import sys

        input = sys.stdin.readline

        s = str(input_data.rstrip("\n"))

        return "Yes" if s[:4] == "YAKI" else "No"

    if __name__ == "__main__":

        slove()
