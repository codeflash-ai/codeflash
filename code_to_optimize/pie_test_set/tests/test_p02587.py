from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02587_0():
    input_content = "3\nba 3\nabc 4\ncbaa 5"
    expected_output = "7"
    run_pie_test_case("../p02587.py", input_content, expected_output)


def test_problem_p02587_1():
    input_content = "2\nabc 1\nab 2"
    expected_output = "-1"
    run_pie_test_case("../p02587.py", input_content, expected_output)


def test_problem_p02587_2():
    input_content = "4\nab 5\ncba 3\na 12\nab 10"
    expected_output = "8"
    run_pie_test_case("../p02587.py", input_content, expected_output)


def test_problem_p02587_3():
    input_content = "2\nabcab 5\ncba 3"
    expected_output = "11"
    run_pie_test_case("../p02587.py", input_content, expected_output)


def test_problem_p02587_4():
    input_content = "3\nba 3\nabc 4\ncbaa 5"
    expected_output = "7"
    run_pie_test_case("../p02587.py", input_content, expected_output)
