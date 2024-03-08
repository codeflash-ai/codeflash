def problem_p03285():
    def judge(a):

        global flg

        if a % 4 == 0 or a % 7 == 0:

            flg = 1

    N = int(eval(input()))

    flg = 0

    judge(N)

    while N > 7:

        N -= 7

        judge(N)

    if flg == 1:

        print("Yes")

    else:

        print("No")


problem_p03285()
