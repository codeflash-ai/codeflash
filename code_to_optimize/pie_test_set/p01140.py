def problem_p01140():
    while 1:

        n, m = list(map(int, input().split(" ")))

        if (n, m) == (0, 0):

            break

        Lon = [0 for i in range(1000 * n + 1)]

        Lat = [0 for i in range(1000 * m + 1)]

        Lon_sum = [0]

        Lat_sum = [0]

        for i in range(0, n):

            tmp_Lon_sum = [0]

            h = int(input())

            for j in Lon_sum:

                Lon[j + h] += 1

                tmp_Lon_sum += [j + h]

            Lon_sum = tmp_Lon_sum

        else:

            Lon_max = Lon_sum[-1]

        for i in range(0, m):

            tmp_Lat_sum = [0]

            w = int(input())

            for j in Lat_sum:

                Lat[j + w] += 1

                tmp_Lat_sum += [j + w]

            Lat_sum = tmp_Lat_sum

        else:

            Lat_max = Lat_sum[-1]

        max_width = min(Lon_max, Lat_max)

        print(sum([Lon[i] * Lat[i] for i in range(1, max_width + 1)]))


problem_p01140()
