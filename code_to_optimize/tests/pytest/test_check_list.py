from code_to_optimize.check_list import check_user_access


def test_check_user_access():
    user_ids = [str(i) for i in range(1000)]
    check_ids = [str(i) for i in range(1000)]
    res = [True] * 1000
    assert check_user_access(user_ids, check_ids) == res

    check_ids = [str(i) for i in range(1000, 2000)]
    res = [False] * 1000
    assert check_user_access(user_ids, check_ids) == res
