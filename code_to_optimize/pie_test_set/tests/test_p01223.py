from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01223_0():
    input_content = (
        "5\n5\n10 70 30 50 90\n2\n20 100\n2\n100 30\n3\n50 50 50\n7\n123 45 678 901 234 567 890"
    )
    expected_output = "60 40\n80 0\n0 70\n0 0\n633 667"
    run_pie_test_case("../p01223.py", input_content, expected_output)


def test_problem_p01223_1():
    input_content = (
        "5\n5\n10 70 30 50 90\n2\n20 100\n2\n100 30\n3\n50 50 50\n7\n123 45 678 901 234 567 890"
    )
    expected_output = "60 40\n80 0\n0 70\n0 0\n633 667"
    run_pie_test_case("../p01223.py", input_content, expected_output)
