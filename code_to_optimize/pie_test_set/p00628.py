def problem_p00628():
    while True:

        number = input().split(" ")

        if number == ["END", "OF", "INPUT"]:
            break

        ans = []

        for num in number:

            ans = ans + ["0"] if num == "" else ans + [str(len(num))]

        print(("".join(ans)))


problem_p00628()
