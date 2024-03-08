def problem_p04043():
    ABC = list(map(int, input().split()))

    if ABC.count(5) == 2 and ABC.count(7):

        print("YES")

    else:

        print("NO")


problem_p04043()
