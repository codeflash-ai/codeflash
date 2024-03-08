from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02614_0():
    input_content = "2 3 2\n..#\n###"
    expected_output = "5"
    run_pie_test_case("../p02614.py", input_content, expected_output)


def test_problem_p02614_1():
    input_content = "2 3 4\n..#"
    expected_output = "1"
    run_pie_test_case("../p02614.py", input_content, expected_output)


def test_problem_p02614_2():
    input_content = "6 6 8\n..##..\n.#..#.\n....#\n\n....#\n....#"
    expected_output = "208"
    run_pie_test_case("../p02614.py", input_content, expected_output)


def test_problem_p02614_3():
    input_content = "2 3 2\n..#"
    expected_output = "5"
    run_pie_test_case("../p02614.py", input_content, expected_output)


def test_problem_p02614_4():
    input_content = "2 2 3"
    expected_output = "0"
    run_pie_test_case("../p02614.py", input_content, expected_output)


def test_problem_p02614_5():
    input_content = "2 3 2\n..#\n###"
    expected_output = "5"
    run_pie_test_case("../p02614.py", input_content, expected_output)
