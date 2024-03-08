from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03030_0():
    input_content = "6\nkhabarovsk 20\nmoscow 10\nkazan 50\nkazan 35\nmoscow 60\nkhabarovsk 40"
    expected_output = "3\n4\n6\n1\n5\n2"
    run_pie_test_case("../p03030.py", input_content, expected_output)


def test_problem_p03030_1():
    input_content = "6\nkhabarovsk 20\nmoscow 10\nkazan 50\nkazan 35\nmoscow 60\nkhabarovsk 40"
    expected_output = "3\n4\n6\n1\n5\n2"
    run_pie_test_case("../p03030.py", input_content, expected_output)


def test_problem_p03030_2():
    input_content = "10\nyakutsk 10\nyakutsk 20\nyakutsk 30\nyakutsk 40\nyakutsk 50\nyakutsk 60\nyakutsk 70\nyakutsk 80\nyakutsk 90\nyakutsk 100"
    expected_output = "10\n9\n8\n7\n6\n5\n4\n3\n2\n1"
    run_pie_test_case("../p03030.py", input_content, expected_output)
