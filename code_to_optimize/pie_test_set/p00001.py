def problem_p00001():
    ans = []

    for i in range(0, 10):

        ans.append(int(eval(input())))

    ans.sort(reverse=True)

    for i in range(0, 3):

        print((ans[i]))


problem_p00001()
