def problem_p00029(input_data):
    x = input_data.split()

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

    return (THEmostWord, longestWord)
