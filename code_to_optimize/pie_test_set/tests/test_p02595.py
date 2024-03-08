from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02595_0():
    input_content = "4 5\n0 5\n-2 4\n3 4\n4 -4"
    expected_output = "3"
    run_pie_test_case("../p02595.py", input_content, expected_output)


def test_problem_p02595_1():
    input_content = "12 3\n1 1\n1 1\n1 1\n1 1\n1 2\n1 3\n2 1\n2 2\n2 3\n3 1\n3 2\n3 3"
    expected_output = "7"
    run_pie_test_case("../p02595.py", input_content, expected_output)


def test_problem_p02595_2():
    input_content = "4 5\n0 5\n-2 4\n3 4\n4 -4"
    expected_output = "3"
    run_pie_test_case("../p02595.py", input_content, expected_output)


def test_problem_p02595_3():
    input_content = "20 100000\n14309 -32939\n-56855 100340\n151364 25430\n103789 -113141\n147404 -136977\n-37006 -30929\n188810 -49557\n13419 70401\n-88280 165170\n-196399 137941\n-176527 -61904\n46659 115261\n-153551 114185\n98784 -6820\n94111 -86268\n-30401 61477\n-55056 7872\n5901 -163796\n138819 -185986\n-69848 -96669"
    expected_output = "6"
    run_pie_test_case("../p02595.py", input_content, expected_output)
