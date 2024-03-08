from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03261_0():
    input_content = "4\nhoge\nenglish\nhoge\nenigma"
    expected_output = "No"
    run_pie_test_case("../p03261.py", input_content, expected_output)


def test_problem_p03261_1():
    input_content = "4\nhoge\nenglish\nhoge\nenigma"
    expected_output = "No"
    run_pie_test_case("../p03261.py", input_content, expected_output)


def test_problem_p03261_2():
    input_content = "9\nbasic\nc\ncpp\nphp\npython\nnadesico\nocaml\nlua\nassembly"
    expected_output = "Yes"
    run_pie_test_case("../p03261.py", input_content, expected_output)


def test_problem_p03261_3():
    input_content = "3\nabc\narc\nagc"
    expected_output = "No"
    run_pie_test_case("../p03261.py", input_content, expected_output)


def test_problem_p03261_4():
    input_content = "8\na\naa\naaa\naaaa\naaaaa\naaaaaa\naaa\naaaaaaa"
    expected_output = "No"
    run_pie_test_case("../p03261.py", input_content, expected_output)
