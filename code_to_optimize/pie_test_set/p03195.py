def problem_p03195():
    def solve():

        for _ in range(int(eval(input()))):

            if int(eval(input())) & 1:

                return "first"

        return "second "

    print((solve()))


problem_p03195()
