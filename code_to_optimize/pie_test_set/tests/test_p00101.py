from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00101_0():
    input_content = "3\nHoshino\nHashino\nMasayuki Hoshino was the grandson of Ieyasu Tokugawa."
    expected_output = "Hoshina\nHashino\nMasayuki Hoshina was the grandson of Ieyasu Tokugawa."
    run_pie_test_case("../p00101.py", input_content, expected_output)


def test_problem_p00101_1():
    input_content = "3\nHoshino\nHashino\nMasayuki Hoshino was the grandson of Ieyasu Tokugawa."
    expected_output = "Hoshina\nHashino\nMasayuki Hoshina was the grandson of Ieyasu Tokugawa."
    run_pie_test_case("../p00101.py", input_content, expected_output)
