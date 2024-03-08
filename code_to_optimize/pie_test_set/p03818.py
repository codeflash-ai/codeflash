def problem_p03818():
    def main():

        N = int(eval(input()))

        A = [int(i) for i in input().split()]

        from collections import Counter

        c = Counter(A)

        even = len([v for v in list(c.values()) if v % 2 == 0])

        ans = len(list(c.keys()))

        if even % 2 == 1:

            ans -= 1

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03818()
