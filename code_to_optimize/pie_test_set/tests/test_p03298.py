from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03298_0():
    input_content = "4\ncabaacba"
    expected_output = "4"
    run_pie_test_case("../p03298.py", input_content, expected_output)


def test_problem_p03298_1():
    input_content = "4\ncabaacba"
    expected_output = "4"
    run_pie_test_case("../p03298.py", input_content, expected_output)


def test_problem_p03298_2():
    input_content = "18\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    expected_output = "9075135300"
    run_pie_test_case("../p03298.py", input_content, expected_output)


def test_problem_p03298_3():
    input_content = "11\nmippiisssisssiipsspiim"
    expected_output = "504"
    run_pie_test_case("../p03298.py", input_content, expected_output)


def test_problem_p03298_4():
    input_content = "4\nabcdefgh"
    expected_output = "0"
    run_pie_test_case("../p03298.py", input_content, expected_output)
