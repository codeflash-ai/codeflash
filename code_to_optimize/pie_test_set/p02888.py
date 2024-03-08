def problem_p02888():
    n = int(eval(input()))

    l = list(map(int, input().split()))

    l.sort()

    l.append(10**9)

    cnt = 0

    i = 0

    # print(l)

    while i < n - 2:

        j = i + 1

        while j < n - 1:

            LL = j + 1

            RR = n - 1

            # print("#",l[i],l[j],l[LL],l[RR])

            while LL <= RR:

                k = (LL + RR) // 2

                if k == n:

                    break

                # print(l)

                # print(i,j,LL,RR,k)

                # print(l[i]+l[j],l[k],l[k+1])

                if l[i] + l[j] > l[k] and l[i] + l[j] <= l[k + 1]:

                    cnt += k - j

                    break

                elif LL == RR:

                    break

                elif l[i] + l[j] <= l[k]:

                    RR = k

                elif l[i] + l[j] > l[k]:

                    LL = k + 1

                elif l[i] + l[j] < l[k] and k == j + 1:

                    break

            j += 1

        i += 1

    print(cnt)


problem_p02888()
