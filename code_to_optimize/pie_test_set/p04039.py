def problem_p04039():
    n, m = input().split()

    k = int(n)

    li = list(input().split())

    for i in range(k, 100000):

        for e in li:

            if e in str(i):

                break

        else:

            print(i)

            break


problem_p04039()
