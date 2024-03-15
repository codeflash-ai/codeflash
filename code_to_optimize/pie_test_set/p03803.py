def problem_p03803(input_data):
    from collections import defaultdict

    def main():

        A, B = list(map(int, input_data.split()))

        d = {1: 14}

        if d.get(A, A) == d.get(B, B):

            return "Draw"

        elif d.get(A, A) > d.get(B, B):

            return "Alice"

        else:

            return "Bob"

    if __name__ == "__main__":

        main()
