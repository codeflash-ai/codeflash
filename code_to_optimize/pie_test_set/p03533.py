def problem_p03533():
    w = [
        "AKIHABARA",
        "KIHABARA",
        "AKIHBARA",
        "AKIHABRA",
        "AKIHABAR",
        "KIHBARA",
        "KIHABRA",
        "KIHABAR",
        "AKIHBRA",
        "AKIHBAR",
        "AKIHABR",
        "KIHBRA",
        "KIHBAR",
        "KIHABR",
        "AKIHBR",
        "KIHBR",
    ]

    s = eval(input())

    ans = 0

    if s in w:
        ans = 1

    print((["NO", "YES"][ans]))


problem_p03533()
