from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00718_0():
    input_content = "10\nxi x9i\ni 9i\nc2x2i 4c8x8i\nm2ci 4m7c9x8i\n9c9x9i i\ni 9m9c9x8i\nm i\ni m\nm9i i\n9m8c7xi c2x8i"
    expected_output = "3x\nx\n6cx\n5m9c9x9i\nm\n9m9c9x9i\nmi\nmi\nmx\n9m9c9x9i"
    run_pie_test_case("../p00718.py", input_content, expected_output)


def test_problem_p00718_1():
    input_content = "10\nxi x9i\ni 9i\nc2x2i 4c8x8i\nm2ci 4m7c9x8i\n9c9x9i i\ni 9m9c9x8i\nm i\ni m\nm9i i\n9m8c7xi c2x8i"
    expected_output = "3x\nx\n6cx\n5m9c9x9i\nm\n9m9c9x9i\nmi\nmi\nmx\n9m9c9x9i"
    run_pie_test_case("../p00718.py", input_content, expected_output)
