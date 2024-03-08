def problem_p03578():
    from collections import Counter

    n = int(eval(input()))

    d = list(map(int, input().split()))

    d_c = Counter(d)

    m = int(eval(input()))

    t = list(map(int, input().split()))

    # print(d_c)

    for i in range(m):

        if t[i] in d_c:

            d_c[t[i]] -= 1

            if d_c[t[i]] < 0:

                print("NO")

                exit()

        else:

            print("NO")

            exit()

    # print(d_c)

    print("YES")


problem_p03578()
