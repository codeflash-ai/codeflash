def problem_p02773():
    import collections

    n = int(eval(input()))

    a = [eval(input()) for _ in range(n)]

    anslist = []

    b = collections.Counter(a)

    c = max(list(b.values()))

    for i in b:

        if b[i] == c:

            anslist.append(i)

    d = sorted(anslist)

    for i in tuple(d):

        print(i)


problem_p02773()
