from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codeflash.code_utils.codeflash_wrap_decorator import (
    _extract_class_name_tracer,
    _get_module_name_cf_tracer,
    extract_test_context_from_frame,
)


@pytest.fixture
def mock_instance():
    mock_obj = Mock()
    mock_obj.__class__.__name__ = "TestClassName"
    return mock_obj


@pytest.fixture
def mock_class():
    mock_cls = Mock()
    mock_cls.__name__ = "TestClassMethod"
    return mock_cls


class TestExtractClassNameTracer:
    
    def test_extract_class_name_with_self(self, mock_instance):
        frame_locals = {"self": mock_instance}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result == "TestClassName"
    
    def test_extract_class_name_with_cls(self, mock_class):
        frame_locals = {"cls": mock_class}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result == "TestClassMethod"
    
    def test_extract_class_name_self_no_class(self, mock_class):
        class NoClassMock:
            @property
            def __class__(self):
                raise AttributeError("no __class__ attribute")
        
        mock_instance = NoClassMock()
        frame_locals = {"self": mock_instance, "cls": mock_class}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result == "TestClassMethod"
    
    def test_extract_class_name_no_self_or_cls(self):
        frame_locals = {"some_var": "value"}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result is None
    
    def test_extract_class_name_exception_handling(self):
        class ExceptionMock:
            @property
            def __class__(self):
                raise Exception("Test exception")
        
        mock_instance = ExceptionMock()
        frame_locals = {"self": mock_instance}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result is None
    
    def test_extract_class_name_with_attribute_error(self):
        class AttributeErrorMock:
            @property
            def __class__(self):
                raise AttributeError("Wrapt-like error")
        
        mock_instance = AttributeErrorMock()
        frame_locals = {"self": mock_instance}
        result = _extract_class_name_tracer(frame_locals)
        
        assert result is None


class TestGetModuleNameCfTracer:
    
    def test_get_module_name_with_valid_frame(self):
        mock_frame = Mock()
        mock_module = Mock()
        mock_module.__name__ = "test_module_name"
        
        with patch("inspect.getmodule", return_value=mock_module):
            result = _get_module_name_cf_tracer(mock_frame)
            assert result == "test_module_name"
    
    def test_get_module_name_from_frame_globals(self):
        mock_frame = Mock()
        mock_frame.f_globals = {"__name__": "module_from_globals"}
        
        with patch("inspect.getmodule", side_effect=Exception("Module not found")):
            result = _get_module_name_cf_tracer(mock_frame)
            assert result == "module_from_globals"
    
    def test_get_module_name_no_name_in_globals(self):
        mock_frame = Mock()
        mock_frame.f_globals = {}
        
        with patch("inspect.getmodule", side_effect=Exception("Module not found")):
            result = _get_module_name_cf_tracer(mock_frame)
            assert result == "unknown_module"
    
    def test_get_module_name_none_frame(self):
        result = _get_module_name_cf_tracer(None)
        assert result == "unknown_module"
    
    def test_get_module_name_module_no_name_attribute(self):
        mock_frame = Mock()
        mock_module = Mock(spec=[])
        mock_frame.f_globals = {"__name__": "fallback_name"}
        
        with patch("inspect.getmodule", return_value=mock_module):
            result = _get_module_name_cf_tracer(mock_frame)
            assert result == "fallback_name"


class TestExtractTestContextFromFrame:
    
    def test_direct_test_function_call(self):
        def test_example_function():
            return extract_test_context_from_frame(Path("/tmp/tests"))
        
        result = test_example_function()
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestExtractTestContextFromFrame"
        assert function_name == "test_example_function"
    
    def test_with_test_class_method(self):
        class TestExampleClass:
            def test_method(self):
                return extract_test_context_from_frame(Path("/tmp/tests"))
        
        instance = TestExampleClass()
        result = instance.test_method()
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestExampleClass"  
        assert function_name == "test_method"
    
    def test_function_without_test_prefix(self):
        result = extract_test_context_from_frame(Path("/tmp/tests"))
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestExtractTestContextFromFrame"
        assert function_name == "test_function_without_test_prefix"
    
    @patch('inspect.currentframe')
    def test_no_test_context_raises_runtime_error(self, mock_current_frame):
        mock_frame = Mock()
        mock_frame.f_back = None
        mock_frame.f_code.co_name = "regular_function"
        mock_frame.f_code.co_filename = "/path/to/regular_file.py"
        mock_frame.f_locals = {}
        mock_frame.f_globals = {"__name__": "regular_module"}
        
        mock_current_frame.return_value = mock_frame
        
        with pytest.raises(RuntimeError, match="No test function found in call stack"):
            extract_test_context_from_frame(Path("/tmp/tests"))
    
    def test_real_call_stack_context(self):
        def nested_function():
            def deeper_function():
                return extract_test_context_from_frame(Path("/tmp/tests"))
            return deeper_function()
        
        result = nested_function()
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestExtractTestContextFromFrame"
        assert function_name == "test_real_call_stack_context"
    


class TestIntegrationScenarios:
    
    def test_pytest_class_method_scenario(self):
        class TestExampleIntegration:
            def test_integration_method(self):
                return extract_test_context_from_frame(Path("/tmp/tests"))
        
        instance = TestExampleIntegration()
        result = instance.test_integration_method()
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestExampleIntegration"
        assert function_name == "test_integration_method"
    
    def test_nested_helper_functions(self):
        def outer_helper():
            def inner_helper():
                def deepest_helper():
                    return extract_test_context_from_frame(Path("/tmp/tests"))
                return deepest_helper()
            return inner_helper()
        
        result = outer_helper()
        module_name, class_name, function_name = result
        
        assert module_name == __name__
        assert class_name == "TestIntegrationScenarios"
        assert function_name == "test_nested_helper_functions"
