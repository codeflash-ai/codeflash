def problem_p00058():
    while True:

        try:

            xA, yA, xB, yB, xC, yC, xD, yD = list(map(float, input().split()))

            if abs((yB - yA) * (yD - yC) + (xB - xA) * (xD - xC)) < 1.0e-12:

                print("YES")

            else:

                print("NO")

        except:

            break


problem_p00058()
