from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02773_0():
    input_content = "7\nbeat\nvet\nbeet\nbed\nvet\nbet\nbeet"
    expected_output = "beet\nvet"
    run_pie_test_case("../p02773.py", input_content, expected_output)


def test_problem_p02773_1():
    input_content = "8\nbuffalo\nbuffalo\nbuffalo\nbuffalo\nbuffalo\nbuffalo\nbuffalo\nbuffalo"
    expected_output = "buffalo"
    run_pie_test_case("../p02773.py", input_content, expected_output)


def test_problem_p02773_2():
    input_content = "7\nbeat\nvet\nbeet\nbed\nvet\nbet\nbeet"
    expected_output = "beet\nvet"
    run_pie_test_case("../p02773.py", input_content, expected_output)


def test_problem_p02773_3():
    input_content = "7\nbass\nbass\nkick\nkick\nbass\nkick\nkick"
    expected_output = "kick"
    run_pie_test_case("../p02773.py", input_content, expected_output)


def test_problem_p02773_4():
    input_content = "4\nushi\ntapu\nnichia\nkun"
    expected_output = "kun\nnichia\ntapu\nushi"
    run_pie_test_case("../p02773.py", input_content, expected_output)
