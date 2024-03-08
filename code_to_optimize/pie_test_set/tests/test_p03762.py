from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03762_0():
    input_content = "3 3\n1 3 4\n1 3 6"
    expected_output = "60"
    run_pie_test_case("../p03762.py", input_content, expected_output)


def test_problem_p03762_1():
    input_content = "3 3\n1 3 4\n1 3 6"
    expected_output = "60"
    run_pie_test_case("../p03762.py", input_content, expected_output)


def test_problem_p03762_2():
    input_content = "6 5\n-790013317 -192321079 95834122 418379342 586260100 802780784\n-253230108 193944314 363756450 712662868 735867677"
    expected_output = "835067060"
    run_pie_test_case("../p03762.py", input_content, expected_output)
