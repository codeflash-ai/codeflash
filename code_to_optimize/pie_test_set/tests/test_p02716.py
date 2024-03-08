from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02716_0():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "12"
    run_pie_test_case("../p02716.py", input_content, expected_output)


def test_problem_p02716_1():
    input_content = "27\n18 -28 18 28 -45 90 -45 23 -53 60 28 -74 -71 35 -26 -62 49 -77 57 24 -70 -93 69 -99 59 57 -49"
    expected_output = "295"
    run_pie_test_case("../p02716.py", input_content, expected_output)


def test_problem_p02716_2():
    input_content = "5\n-1000 -100 -10 0 10"
    expected_output = "0"
    run_pie_test_case("../p02716.py", input_content, expected_output)


def test_problem_p02716_3():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "12"
    run_pie_test_case("../p02716.py", input_content, expected_output)


def test_problem_p02716_4():
    input_content = "10\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "5000000000"
    run_pie_test_case("../p02716.py", input_content, expected_output)
