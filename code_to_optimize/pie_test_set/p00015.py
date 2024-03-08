def problem_p00015():
    import math

    n = int(eval(input()))

    for i in range(n):

        num1 = int(eval(input()))

        num2 = int(eval(input()))

        total = num1 + num2

        if len(str(total)) > 80:

            print("overflow")

        else:

            print(total)


problem_p00015()
