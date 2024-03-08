def problem_p03807():
    n = int(input())

    a = list(map(int, input().split()))

    even, odd = 0, 0

    for i in a:

        if i % 2 == 0:

            even += 1

        else:

            odd += 1

    if odd == 0:

        print("YES")

    elif even == 0:

        print("YES") if odd % 2 == 0 else print("NO")

    else:

        if odd == 1 and even == 1:

            print("NO")

        else:

            print("YES") if odd % 2 == 0 else print("NO")


problem_p03807()
