def problem_p02431():
    # AOJ ITP2_1_A: Vector

    # Python3 2018.6.24 bal4u

    from collections import deque

    Q = deque()

    q = int(eval(input()))

    for i in range(q):

        a = eval(input())

        if a[0] == "2":
            Q.pop()  # popBack

        else:

            id, v = list(map(int, a.split()))

            if id == 0:
                Q.append(v)  # pushBack

            else:
                print((Q[v]))  # randomAccess


problem_p02431()
