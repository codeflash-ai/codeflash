def problem_p00029():
    x = input().split()

    longestWord = ""

    mostWord = 0

    currentCount = 0

    THEmostWord = ""

    for r in range(len(x)):

        if len(x[r]) > len(longestWord):

            longestWord = x[r]

    for j in range(len(x)):

        thing = x[j]

        currentCount = x.count(thing)

        if currentCount > mostWord:

            mostWord = currentCount

            THEmostWord = thing

    print((THEmostWord, longestWord))


problem_p00029()
