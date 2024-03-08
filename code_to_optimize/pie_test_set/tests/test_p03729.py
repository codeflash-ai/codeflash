from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03729_0():
    input_content = "rng gorilla apple"
    expected_output = "YES"
    run_pie_test_case("../p03729.py", input_content, expected_output)


def test_problem_p03729_1():
    input_content = "a a a"
    expected_output = "YES"
    run_pie_test_case("../p03729.py", input_content, expected_output)


def test_problem_p03729_2():
    input_content = "rng gorilla apple"
    expected_output = "YES"
    run_pie_test_case("../p03729.py", input_content, expected_output)


def test_problem_p03729_3():
    input_content = "aaaaaaaaab aaaaaaaaaa aaaaaaaaab"
    expected_output = "NO"
    run_pie_test_case("../p03729.py", input_content, expected_output)


def test_problem_p03729_4():
    input_content = "yakiniku unagi sushi"
    expected_output = "NO"
    run_pie_test_case("../p03729.py", input_content, expected_output)
