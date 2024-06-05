def problem_p03285(input_data):
    def judge(a):

        global flg

        if a % 4 == 0 or a % 7 == 0:

            flg = 1

    N = int(eval(input_data))

    flg = 0

    judge(N)

    while N > 7:

        N -= 7

        judge(N)

    if flg == 1:

        return "Yes"

    else:

        return "No"
