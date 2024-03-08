from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01646_0():
    input_content = (
        "4\ncba\ncab\nb\na\n3\nbca\nab\na\n5\nabc\nacb\nb\nc\nc\n5\nabc\nacb\nc\nb\nb\n0"
    )
    expected_output = "yes\nno\nyes\nno"
    run_pie_test_case("../p01646.py", input_content, expected_output)


def test_problem_p01646_1():
    input_content = (
        "4\ncba\ncab\nb\na\n3\nbca\nab\na\n5\nabc\nacb\nb\nc\nc\n5\nabc\nacb\nc\nb\nb\n0"
    )
    expected_output = "yes\nno\nyes\nno"
    run_pie_test_case("../p01646.py", input_content, expected_output)
