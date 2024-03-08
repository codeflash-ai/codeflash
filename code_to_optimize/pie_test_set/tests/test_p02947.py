from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02947_0():
    input_content = "3\nacornistnt\npeanutbomb\nconstraint"
    expected_output = "1"
    run_pie_test_case("../p02947.py", input_content, expected_output)


def test_problem_p02947_1():
    input_content = "5\nabaaaaaaaa\noneplustwo\naaaaaaaaba\ntwoplusone\naaaabaaaaa"
    expected_output = "4"
    run_pie_test_case("../p02947.py", input_content, expected_output)


def test_problem_p02947_2():
    input_content = "2\noneplustwo\nninemodsix"
    expected_output = "0"
    run_pie_test_case("../p02947.py", input_content, expected_output)


def test_problem_p02947_3():
    input_content = "3\nacornistnt\npeanutbomb\nconstraint"
    expected_output = "1"
    run_pie_test_case("../p02947.py", input_content, expected_output)
