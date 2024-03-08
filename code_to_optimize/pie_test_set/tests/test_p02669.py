from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02669_0():
    input_content = "5\n11 1 2 4 8\n11 1 2 2 8\n32 10 8 5 4\n29384293847243 454353412 332423423 934923490 1\n900000000000000000 332423423 454353412 934923490 987654321"
    expected_output = "20\n19\n26\n3821859835\n23441258666"
    run_pie_test_case("../p02669.py", input_content, expected_output)


def test_problem_p02669_1():
    input_content = "5\n11 1 2 4 8\n11 1 2 2 8\n32 10 8 5 4\n29384293847243 454353412 332423423 934923490 1\n900000000000000000 332423423 454353412 934923490 987654321"
    expected_output = "20\n19\n26\n3821859835\n23441258666"
    run_pie_test_case("../p02669.py", input_content, expected_output)
