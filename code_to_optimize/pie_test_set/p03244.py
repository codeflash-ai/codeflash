def problem_p03244():
    from collections import Counter

    n = int(eval(input()))

    lst_v = list(map(int, input().split()))

    if len(set(lst_v)) == 1:

        ans = n // 2

    else:

        lst_odd = lst_v[::2]

        lst_even = lst_v[1::2]

        cnt_odd = sorted(list(Counter(lst_odd).items()), key=lambda x: -x[1])

        cnt_even = sorted(list(Counter(lst_even).items()), key=lambda x: -x[1])

        if cnt_odd[0][0] == cnt_even[0][0]:

            rem_number = max(cnt_odd[0][1] + cnt_even[1][1], cnt_odd[1][1] + cnt_even[0][1])

        else:

            rem_number = cnt_odd[0][1] + cnt_even[0][1]

        sum_odd = sum(t[1] for t in cnt_odd)

        sum_even = sum(t[1] for t in cnt_even)

        ans = sum_odd + sum_even - rem_number

    print(ans)


problem_p03244()
