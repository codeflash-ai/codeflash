def problem_p00033():
    n = eval(input())

    for i in range(n):

        dt = list(map(int, input().split()))

        a1 = 0

        a2 = 0

        for e in dt:

            if e > a1:
                a1 = e

            elif e > a2:
                a2 = e

            else:

                print("NO")

                break

        else:

            print("YES")


problem_p00033()
