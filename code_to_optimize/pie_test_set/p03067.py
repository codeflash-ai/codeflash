def problem_p03067():
    a, b, c = list(map(int, input().split()))

    if a < c < b or b < c < a:

        print("Yes")

    else:

        print("No")


problem_p03067()
