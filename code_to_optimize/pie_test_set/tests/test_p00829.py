from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00829_0():
    input_content = "8\n1 1 1 1 1 1 1 1 8\n3 2 3 2 3 2 3 2 6\n3 4 4 7 7 b a 2 2e\ne1 13 ce 28 ca 6 ab 46 a6d\nb08 49e2 6128 f27 8cf2 bc50 7380 7fe1 723b\n4eba eb4 a352 fd14 6ac1 eed1 dd06 bb83 392bc\nef593c08 847e522f 74c02b9c 26f3a4e1 e2720a01 6fe66007\n7a4e96ad 6ee5cef6 3853cd88\n60202fb8 757d6d66 9c3a9525 fbcd7983 82b9571c ddc54bab 853e52da\n22047c88 e5524401"
    expected_output = "0\n2\n6\n1c6\n4924afc7\nffff95c5\n546991d\n901c4a16"
    run_pie_test_case("../p00829.py", input_content, expected_output)


def test_problem_p00829_1():
    input_content = "8\n1 1 1 1 1 1 1 1 8\n3 2 3 2 3 2 3 2 6\n3 4 4 7 7 b a 2 2e\ne1 13 ce 28 ca 6 ab 46 a6d\nb08 49e2 6128 f27 8cf2 bc50 7380 7fe1 723b\n4eba eb4 a352 fd14 6ac1 eed1 dd06 bb83 392bc\nef593c08 847e522f 74c02b9c 26f3a4e1 e2720a01 6fe66007\n7a4e96ad 6ee5cef6 3853cd88\n60202fb8 757d6d66 9c3a9525 fbcd7983 82b9571c ddc54bab 853e52da\n22047c88 e5524401"
    expected_output = "0\n2\n6\n1c6\n4924afc7\nffff95c5\n546991d\n901c4a16"
    run_pie_test_case("../p00829.py", input_content, expected_output)
