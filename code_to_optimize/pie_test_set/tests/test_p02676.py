from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02676_0():
    input_content = "7\nnikoandsolstice"
    expected_output = "nikoand..."
    run_pie_test_case("../p02676.py", input_content, expected_output)


def test_problem_p02676_1():
    input_content = "7\nnikoandsolstice"
    expected_output = "nikoand..."
    run_pie_test_case("../p02676.py", input_content, expected_output)


def test_problem_p02676_2():
    input_content = "40\nferelibenterhominesidquodvoluntcredunt"
    expected_output = "ferelibenterhominesidquodvoluntcredunt"
    run_pie_test_case("../p02676.py", input_content, expected_output)
