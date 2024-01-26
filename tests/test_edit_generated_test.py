import icontract_hypothesis
from hypothesis import given, strategies as st

from django.aiservice.injectperf.edit_generated_test import parse_module_to_cst, is_parsable


# Test that the function does not raise any error for non-empty strings


@given(module_str=st.text().filter(lambda module_str: is_parsable(module_str)))
def test_parse_module_to_cst_parsable_string(module_str: str):
    parse_module_to_cst(module_str)


# Automatically generate tests based on icontract decorators


icontract_hypothesis.test_with_inferred_strategy(parse_module_to_cst)
