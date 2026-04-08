"""Test that property getters are excluded from optimization.

Property getters defined via Object.defineProperty should not be
optimized because they're not directly callable and tests cannot
access them by the function name.

Relates to bug: Generated tests try to call getters directly
(e.g., `obj.getterFunc()`) when they should access the property
(e.g., `obj.propertyName`).
"""

from pathlib import Path

from codeflash.discovery.functions_to_optimize import find_all_functions_in_file


class TestPropertyGetterExclusion:
    """Tests for excluding property getters from function discovery."""

    def test_object_define_property_getter_excluded(self, tmp_path: Path) -> None:
        """Test that functions used as property getters are excluded.

        When a function is defined as `get: function foo() {...}` inside
        Object.defineProperty, it should not be discovered as an optimizable
        function because:
        1. It's not directly accessible by the function name
        2. Generated tests would fail trying to call it directly
        3. Property access patterns are different from function calls

        This reproduces the Express.js pattern where getrouter is defined
        as a property getter inside the init function.
        """
        js_file = tmp_path / "app.js"
        js_file.write_text("""
const app = {};

// Express pattern: getter nested inside a function
app.init = function init() {
  var router = null;

  // Property getter pattern (like express application.js line 72)
  Object.defineProperty(this, 'router', {
    configurable: true,
    get: function getrouter() {
      if (router === null) {
        router = { use: () => {} };
      }
      return router;
    }
  });
};

// Normal exported function (should be found)
export function normalFunction() {
  return 42;
}

module.exports = app;
""")

        functions = find_all_functions_in_file(js_file)
        function_names = {fn.function_name for fn in functions.get(js_file, [])}

        # Property getter should NOT be found
        assert "getrouter" not in function_names, (
            "Property getter 'getrouter' should be excluded from optimization. "
            "Tests cannot access it as init.getrouter() - they would need to access "
            "the 'router' property via an instance instead."
        )

        # Normal exported function should be found
        assert "normalFunction" in function_names

    def test_object_define_property_setter_excluded(self, tmp_path: Path) -> None:
        """Test that functions used as property setters are also excluded."""
        js_file = tmp_path / "app.js"
        js_file.write_text("""
const app = {};

Object.defineProperty(app, 'value', {
  set: function setvalue(val) {
    this._value = val;
  },
  get: function getvalue() {
    return this._value;
  }
});

export function helper() {
  return 1;
}
""")

        functions = find_all_functions_in_file(js_file)
        function_names = {fn.function_name for fn in functions.get(js_file, [])}

        # Neither getter nor setter should be found
        assert "setvalue" not in function_names
        assert "getvalue" not in function_names

        # Helper function should still be found
        assert "helper" in function_names

    def test_object_literal_getter_excluded(self, tmp_path: Path) -> None:
        """Test that getter methods in object literals are excluded."""
        js_file = tmp_path / "obj.js"
        js_file.write_text("""
const obj = {
  get router() {
    return this._router;
  },

  // Regular method should be excluded too (it's in an object literal)
  method() {
    return 1;
  }
};

export function exported() {
  return obj;
}
""")

        functions = find_all_functions_in_file(js_file)
        function_names = {fn.function_name for fn in functions.get(js_file, [])}

        # Getter in object literal should not be found
        assert "router" not in function_names

        # Regular method in object literal should also not be found
        # (per existing code logic)
        assert "method" not in function_names

        # Exported function should be found
        assert "exported" in function_names
