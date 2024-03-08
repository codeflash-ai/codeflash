def problem_p00613():
    while 1:

        k = eval(input()) - 1

        if not k + 1:
            break

        print(sum(map(int, input().split())) / k)


problem_p00613()
