def problem_p03693():
    r, g, b = list(map(int, input().split()))

    if (10 * g + b) % 4 == 0:

        print("YES")

    else:

        print("NO")


problem_p03693()
