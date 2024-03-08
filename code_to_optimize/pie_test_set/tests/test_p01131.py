from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01131_0():
    input_content = "5\n20\n220\n222220\n44033055505550666011011111090666077705550301110\n000555555550000330000444000080000200004440000"
    expected_output = "a\nb\nb\nhello, world!\nkeitai"
    run_pie_test_case("../p01131.py", input_content, expected_output)


def test_problem_p01131_1():
    input_content = "5\n20\n220\n222220\n44033055505550666011011111090666077705550301110\n000555555550000330000444000080000200004440000"
    expected_output = "a\nb\nb\nhello, world!\nkeitai"
    run_pie_test_case("../p01131.py", input_content, expected_output)
