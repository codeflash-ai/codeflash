def problem_p03803():
    from collections import defaultdict

    def main():

        A, B = list(map(int, input().split()))

        d = {1: 14}

        if d.get(A, A) == d.get(B, B):

            print("Draw")

        elif d.get(A, A) > d.get(B, B):

            print("Alice")

        else:

            print("Bob")

    if __name__ == "__main__":

        main()


problem_p03803()
