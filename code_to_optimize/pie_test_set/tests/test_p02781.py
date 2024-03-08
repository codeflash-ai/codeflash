from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02781_0():
    input_content = "100\n1"
    expected_output = "19"
    run_pie_test_case("../p02781.py", input_content, expected_output)


def test_problem_p02781_1():
    input_content = "9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999\n3"
    expected_output = "117879300"
    run_pie_test_case("../p02781.py", input_content, expected_output)


def test_problem_p02781_2():
    input_content = "314159\n2"
    expected_output = "937"
    run_pie_test_case("../p02781.py", input_content, expected_output)


def test_problem_p02781_3():
    input_content = "25\n2"
    expected_output = "14"
    run_pie_test_case("../p02781.py", input_content, expected_output)


def test_problem_p02781_4():
    input_content = "100\n1"
    expected_output = "19"
    run_pie_test_case("../p02781.py", input_content, expected_output)
