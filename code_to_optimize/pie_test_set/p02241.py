def problem_p02241():
    n = int(eval(input()))

    edgeList = []

    rootList = [-1 for i in range(n)]

    def getRoot(x):

        r = rootList[x]

        if r < 0:

            rootList[x] = x

        elif r != x:

            rootList[x] = getRoot(r)

        return rootList[x]

    for i in range(n):

        a = list(map(int, input().split()))

        for j in range(i):

            if a[j] != -1:

                edgeList.append([a[j], getRoot(i), getRoot(j)])

    sumLength = 0

    edgeList.sort(key=lambda x: x[0])

    for e in edgeList:

        x = getRoot(e[1])

        y = getRoot(e[2])

        if x != y:

            sumLength += e[0]

            rootList[x] = rootList[y] = min(x, y)

    print(sumLength)


problem_p02241()
