def problem_p00093():
    def get_data():

        while True:

            a, b = list(map(int, input().split()))

            if a == b == 0:

                break

            yield a, b

    input_data = list(get_data())

    for i, data in enumerate(input_data):

        a, b = data

        flg = False

        for x in range(a, b + 1):

            if (x % 4 == 0 and x % 100 != 0) or x % 400 == 0:

                if not flg:

                    flg = True

                print(x)

        if not flg:

            print("NA")

        if i != len(input_data) - 1:

            print()


problem_p00093()
