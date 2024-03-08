def problem_p01202():
    for i in range(eval(input())):

        step = input()

        flag = 0

        for j in range(len(step) - 1):

            if step[j] == step[j + 1]:

                print("No")

                break

        else:

            print("Yes")


problem_p01202()
