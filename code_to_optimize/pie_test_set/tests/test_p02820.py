from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02820_0():
    input_content = "5 2\n8 7 6\nrsrpr"
    expected_output = "27"
    run_pie_test_case("../p02820.py", input_content, expected_output)


def test_problem_p02820_1():
    input_content = "5 2\n8 7 6\nrsrpr"
    expected_output = "27"
    run_pie_test_case("../p02820.py", input_content, expected_output)


def test_problem_p02820_2():
    input_content = "7 1\n100 10 1\nssssppr"
    expected_output = "211"
    run_pie_test_case("../p02820.py", input_content, expected_output)


def test_problem_p02820_3():
    input_content = "30 5\n325 234 123\nrspsspspsrpspsppprpsprpssprpsr"
    expected_output = "4996"
    run_pie_test_case("../p02820.py", input_content, expected_output)
