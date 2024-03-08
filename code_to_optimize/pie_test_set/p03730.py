def problem_p03730():
    a, b, c = list(map(int, input().split()))

    for i in range(1, 100000):

        if a * i % b == c:

            print("YES")

            exit()

    print("NO")


problem_p03730()
