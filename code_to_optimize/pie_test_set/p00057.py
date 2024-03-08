def problem_p00057():
    while True:

        try:

            n = int(input())

            area = 2

            for i in range(n - 1):

                area += i + 2

            print(area)

        except:

            break


problem_p00057()
