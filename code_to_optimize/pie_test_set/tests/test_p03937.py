from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03937_0():
    input_content = "4 5\n##...\n.##..\n..##.\n...##"
    expected_output = "Possible"
    run_pie_test_case("../p03937.py", input_content, expected_output)


def test_problem_p03937_1():
    input_content = "5 3\n\n..#\n\n.."
    expected_output = "Impossible"
    run_pie_test_case("../p03937.py", input_content, expected_output)


def test_problem_p03937_2():
    input_content = "4 5\n##...\n.##..\n..##.\n...##"
    expected_output = "Possible"
    run_pie_test_case("../p03937.py", input_content, expected_output)


def test_problem_p03937_3():
    input_content = "4 5\n...\n.###.\n.###.\n...##"
    expected_output = "Impossible"
    run_pie_test_case("../p03937.py", input_content, expected_output)


def test_problem_p03937_4():
    input_content = "4 5\n...\n.##..\n..##.\n...##"
    expected_output = "Possible"
    run_pie_test_case("../p03937.py", input_content, expected_output)
