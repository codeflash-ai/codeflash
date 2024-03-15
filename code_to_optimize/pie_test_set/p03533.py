def problem_p03533(input_data):
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

    s = eval(input_data)

    ans = 0

    if s in w:
        ans = 1

    return ["NO", "YES"][ans]
