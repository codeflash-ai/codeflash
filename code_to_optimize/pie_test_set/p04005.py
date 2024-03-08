def problem_p04005():
    """

    ある平面のブロック数×(0,1)

    一つでも偶数があれば、均等分割可能

    すべて奇数のとき、

    ある平面のブロック数を最小となるようにすると、その数が答えになる

    """

    arr = list(map(int, input().split()))

    ret = -1

    if 0 in list([x % 2 for x in arr]):

        ret = 0

    else:

        arr_sorted = sorted(arr)

        ret = arr_sorted[0] * arr_sorted[1]

    print(ret)


problem_p04005()
