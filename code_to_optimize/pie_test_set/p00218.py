def problem_p00218():
    while True:

        n = int(eval(input()))

        if n == 0:

            break

        for _ in range(n):

            p = list(map(int, input().split()))

            if 100 in p:

                print("A")

            elif (p[0] + p[1]) // 2 >= 90:

                print("A")

            elif sum(p) // 3 >= 80:

                print("A")

            elif sum(p) // 3 >= 70:

                print("B")

            elif sum(p) // 3 >= 50 and (p[0] >= 80 or p[1] >= 80):

                print("B")

            else:

                print("C")


problem_p00218()
