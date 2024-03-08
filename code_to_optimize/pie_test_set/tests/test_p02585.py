from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02585_0():
    input_content = "5 2\n2 4 5 1 3\n3 4 -10 -8 8"
    expected_output = "8"
    run_pie_test_case("../p02585.py", input_content, expected_output)


def test_problem_p02585_1():
    input_content = "10 58\n9 1 6 7 8 4 3 2 10 5\n695279662 988782657 -119067776 382975538 -151885171 -177220596 -169777795 37619092 389386780 980092719"
    expected_output = "29507023469"
    run_pie_test_case("../p02585.py", input_content, expected_output)


def test_problem_p02585_2():
    input_content = "3 3\n3 1 2\n-1000 -2000 -3000"
    expected_output = "-1000"
    run_pie_test_case("../p02585.py", input_content, expected_output)


def test_problem_p02585_3():
    input_content = "5 2\n2 4 5 1 3\n3 4 -10 -8 8"
    expected_output = "8"
    run_pie_test_case("../p02585.py", input_content, expected_output)


def test_problem_p02585_4():
    input_content = "2 3\n2 1\n10 -7"
    expected_output = "13"
    run_pie_test_case("../p02585.py", input_content, expected_output)
