def problem_p03355():
    # 解説

    s = eval(input())

    k = int(eval(input()))

    st = set()

    for start in range(len(s)):

        for len_ in range(1, k + 1):

            st.add(s[start : start + len_])

    (*a,) = sorted(st)

    print((a[k - 1]))

    # 高々k文字

    # kが小さいので

    # 全列挙してソート


problem_p03355()
