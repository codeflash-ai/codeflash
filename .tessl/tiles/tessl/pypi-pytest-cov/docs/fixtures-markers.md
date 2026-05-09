# Fixtures and Markers

pytest-cov provides built-in pytest fixtures and markers that allow fine-grained control over coverage measurement at the test level. These tools enable developers to exclude specific tests from coverage or access the underlying coverage object for advanced scenarios.

## Capabilities

### Coverage Control Fixtures

Fixtures that control coverage behavior during test execution.

```python { .api }
@pytest.fixture
def no_cover():
    """
    A pytest fixture to disable coverage for specific tests.
    
    When used as a fixture parameter, disables coverage measurement
    for the duration of the test function. The fixture itself does
    nothing - its presence signals the coverage system to pause
    measurement.
    
    Usage:
        def test_excluded_function(no_cover):
            # This test will not be included in coverage reports
            pass
    """

@pytest.fixture  
def cov(request):
    """
    A pytest fixture to provide access to the underlying coverage object.
    
    Returns the active coverage.Coverage instance if coverage is enabled,
    allowing tests to inspect coverage state, access configuration, or
    perform advanced coverage operations.
    
    Args:
        request: pytest request object
        
    Returns:
        coverage.Coverage: Active coverage instance, or None if disabled
        
    Usage:
        def test_coverage_info(cov):
            if cov:
                print(f"Data file: {cov.config.data_file}")
                print(f"Source: {cov.config.source}")
    """
```

**Usage Examples:**

```python
import pytest

def test_normal_coverage():
    """This test will be included in coverage measurement."""
    import mymodule
    assert mymodule.function() == "expected"

def test_excluded_with_fixture(no_cover):
    """This test will be excluded from coverage."""
    import mymodule
    # This code execution won't count toward coverage
    mymodule.debug_function()

def test_coverage_access(cov):
    """Test that can access coverage configuration."""
    if cov:
        # Can inspect coverage settings
        assert cov.config.data_file
        assert hasattr(cov.config, 'source')
    
    # Test code still gets measured for coverage
    import mymodule
    mymodule.function()

def test_both_fixtures(cov, no_cover):
    """Test using both fixtures - coverage access but no measurement."""
    if cov:
        print(f"Coverage enabled but test excluded")
    # This test code won't be measured for coverage
```

### Coverage Control Markers

pytest markers that provide declarative coverage control.

```python { .api }
# Marker registration (done automatically by plugin)
config.addinivalue_line('markers', 'no_cover: disable coverage for this test.')
```

The `no_cover` marker provides an alternative to the fixture for excluding tests from coverage measurement.

**Usage Examples:**

```python
import pytest

@pytest.mark.no_cover  
def test_excluded_with_marker():
    """This test will be excluded from coverage measurement."""
    import mymodule
    # Code execution here won't count toward coverage
    mymodule.debug_function()

@pytest.mark.no_cover
def test_debugging_code():
    """Useful for tests that exercise debugging or development-only code."""
    import mymodule
    mymodule.internal_debug_function()
    
class TestDevelopmentFeatures:
    """Test class for development features."""
    
    @pytest.mark.no_cover
    def test_dev_feature_a(self):
        """Development feature test excluded from coverage."""
        pass
        
    def test_production_feature(self):
        """Production feature test included in coverage."""
        pass
```

### Fixture Implementation Details

Internal implementation of the fixture system and coverage control.

```python { .api }
def cov(request):
    """
    Implementation of the cov fixture.
    
    Checks if the '_cov' plugin is registered and active, then
    returns the coverage controller's coverage instance if available.
    
    Returns None if:
    - Coverage plugin is not registered
    - Coverage plugin has no active controller  
    - Coverage controller has no coverage instance
    """
```

The fixture implementation:

1. **Plugin Detection**: Uses `pluginmanager.hasplugin('_cov')` to check if coverage is active
2. **Controller Access**: Gets the `CovPlugin` instance via `pluginmanager.getplugin('_cov')`
3. **Coverage Instance**: Returns `plugin.cov_controller.cov` if available
4. **Graceful Fallback**: Returns `None` if any step fails

### Marker Processing

How the coverage system processes markers and fixtures during test execution.

```python { .api }  
def pytest_runtest_call(self, item):
    """
    Process coverage control markers and fixtures during test execution.
    
    Checks for:
    - @pytest.mark.no_cover marker on test function
    - 'no_cover' in test's fixture names
    
    If either is found, pauses coverage before test execution
    and resumes coverage after test completion.
    """
```

The marker/fixture processing logic:

```python
# Check for marker or fixture
if item.get_closest_marker('no_cover') or 'no_cover' in getattr(item, 'fixturenames', ()):
    self.cov_controller.pause()
    yield  # Execute test
    self.cov_controller.resume()
else:
    yield  # Execute test normally with coverage
```

## Integration Patterns

### Combining Fixtures and Markers

Different ways to control coverage in test scenarios:

```python
# Using marker (preferred for permanent exclusions)
@pytest.mark.no_cover
def test_debug_helper():
    pass

# Using fixture (useful for conditional logic)  
def test_conditional_coverage(no_cover, some_condition):
    if some_condition:
        # Test logic that should be excluded
        pass

# Accessing coverage while excluding test
@pytest.mark.no_cover
def test_coverage_inspection(cov):
    if cov:
        print(f"Coverage active but test excluded")
```

### Conditional Coverage Control

Using fixtures for dynamic coverage control:

```python
@pytest.fixture  
def maybe_no_cover(request):
    """Conditionally disable coverage based on test conditions."""
    if request.config.getoption("--debug-mode"):
        request.getfixturevalue("no_cover")

def test_with_conditional_coverage(maybe_no_cover):
    """Coverage depends on command-line flag."""
    # Test code here
    pass
```

### Coverage Inspection Patterns

Common patterns for using the `cov` fixture:

```python
def test_coverage_configuration(cov): 
    """Inspect coverage configuration."""
    if cov:
        assert cov.config.branch is True
        assert cov.config.source == ['mypackage']

def test_coverage_data_access(cov):
    """Access coverage measurement data.""" 
    if cov:
        # Get current coverage data
        data = cov.get_data()
        measured_files = data.measured_files()
        assert len(measured_files) > 0

def test_coverage_reporting(cov, tmp_path):
    """Generate custom coverage reports."""
    if cov:
        # Generate custom report
        report_file = tmp_path / "custom_report.txt"
        with open(report_file, 'w') as f:
            cov.report(file=f)
        assert report_file.exists()
```

### Test Organization

Organizing tests with coverage control:

```python
class TestProductionCode:
    """Tests for production functionality - all measured."""
    
    def test_feature_a(self):
        import mymodule
        assert mymodule.feature_a() 
        
    def test_feature_b(self):
        import mymodule
        assert mymodule.feature_b()

class TestDevelopmentCode:  
    """Tests for development/debug code - excluded from coverage."""
    
    @pytest.mark.no_cover
    def test_debug_mode(self):
        import mymodule
        mymodule.enable_debug_mode()
        
    @pytest.mark.no_cover  
    def test_development_feature(self):
        import mymodule
        mymodule.experimental_feature()

class TestCoverageAware:
    """Tests that need coverage introspection."""
    
    def test_with_coverage_check(self, cov):
        """Test that verifies coverage is working."""
        if cov:
            # Verify coverage is measuring our module
            import mymodule
            mymodule.function()
            
            data = cov.get_data()
            assert any('mymodule' in f for f in data.measured_files())
```