def problem_p00083():
    while True:

        try:

            y, m, d = list(map(int, input().split()))

            ymd = 10000 * y + 100 * m + d

            if ymd < 18680908:

                print("pre-meiji")

            elif ymd < 19120730:

                print("meiji", y - 1867, m, d)

            elif ymd < 19261225:

                print("taisho", y - 1911, m, d)

            elif ymd < 19890108:

                print("showa", y - 1925, m, d)

            else:

                print("heisei", y - 1988, m, d)

        except:

            break


problem_p00083()
