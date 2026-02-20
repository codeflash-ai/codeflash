"""Tests for Java coverage utilities (JaCoCo integration)."""

from pathlib import Path

from codeflash.languages.java.build_tools import (
    JACOCO_PLUGIN_VERSION,
    add_jacoco_plugin_to_pom,
    get_jacoco_xml_path,
    is_jacoco_configured,
)
from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown, CoverageStatus, FunctionSource
from codeflash.verification.coverage_utils import JacocoCoverageUtils


def create_mock_code_context(helper_functions: list[FunctionSource] | None = None) -> CodeOptimizationContext:
    """Create a minimal mock CodeOptimizationContext for testing."""
    empty_markdown = CodeStringsMarkdown(code_strings=[], language="java")
    return CodeOptimizationContext(
        testgen_context=empty_markdown,
        read_writable_code=empty_markdown,
        read_only_context_code="",
        hashing_code_context="",
        hashing_code_context_hash="",
        helper_functions=helper_functions or [],
        preexisting_objects=set(),
    )


def make_function_source(only_function_name: str, qualified_name: str, file_path: Path) -> FunctionSource:
    return FunctionSource(
        file_path=file_path,
        qualified_name=qualified_name,
        fully_qualified_name=qualified_name,
        only_function_name=only_function_name,
        source_code="",
    )


# Sample JaCoCo XML report for testing
SAMPLE_JACOCO_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE report PUBLIC "-//JACOCO//DTD Report 1.1//EN" "report.dtd">
<report name="example">
    <sessioninfo id="test-session" start="1704067200000" dump="1704070800000"/>
    <package name="com/example">
        <class name="com/example/Calculator" sourcefilename="Calculator.java">
            <method name="&lt;init&gt;" desc="(I)V" line="19">
                <counter type="INSTRUCTION" missed="0" covered="8"/>
                <counter type="LINE" missed="0" covered="4"/>
                <counter type="COMPLEXITY" missed="0" covered="1"/>
                <counter type="METHOD" missed="0" covered="1"/>
            </method>
            <method name="add" desc="(DD)D" line="40">
                <counter type="INSTRUCTION" missed="0" covered="5"/>
                <counter type="BRANCH" missed="0" covered="0"/>
                <counter type="LINE" missed="0" covered="2"/>
                <counter type="COMPLEXITY" missed="0" covered="1"/>
                <counter type="METHOD" missed="0" covered="1"/>
            </method>
            <method name="subtract" desc="(DD)D" line="50">
                <counter type="INSTRUCTION" missed="3" covered="0"/>
                <counter type="BRANCH" missed="0" covered="0"/>
                <counter type="LINE" missed="2" covered="0"/>
                <counter type="COMPLEXITY" missed="1" covered="0"/>
                <counter type="METHOD" missed="1" covered="0"/>
            </method>
            <method name="multiply" desc="(DD)D" line="60">
                <counter type="INSTRUCTION" missed="0" covered="4"/>
                <counter type="BRANCH" missed="1" covered="1"/>
                <counter type="LINE" missed="0" covered="3"/>
                <counter type="COMPLEXITY" missed="1" covered="1"/>
                <counter type="METHOD" missed="0" covered="1"/>
            </method>
            <counter type="INSTRUCTION" missed="3" covered="17"/>
            <counter type="BRANCH" missed="1" covered="1"/>
            <counter type="LINE" missed="2" covered="9"/>
            <counter type="COMPLEXITY" missed="2" covered="3"/>
            <counter type="METHOD" missed="1" covered="3"/>
            <counter type="CLASS" missed="0" covered="1"/>
        </class>
        <sourcefile name="Calculator.java">
            <line nr="19" mi="0" ci="2" mb="0" cb="0"/>
            <line nr="20" mi="0" ci="3" mb="0" cb="0"/>
            <line nr="21" mi="0" ci="2" mb="0" cb="0"/>
            <line nr="22" mi="0" ci="1" mb="0" cb="0"/>
            <line nr="40" mi="0" ci="3" mb="0" cb="0"/>
            <line nr="41" mi="0" ci="2" mb="0" cb="0"/>
            <line nr="50" mi="2" ci="0" mb="0" cb="0"/>
            <line nr="51" mi="1" ci="0" mb="0" cb="0"/>
            <line nr="60" mi="0" ci="2" mb="1" cb="1"/>
            <line nr="61" mi="0" ci="1" mb="0" cb="0"/>
            <line nr="62" mi="0" ci="1" mb="0" cb="0"/>
            <counter type="INSTRUCTION" missed="3" covered="17"/>
            <counter type="BRANCH" missed="1" covered="1"/>
            <counter type="LINE" missed="2" covered="9"/>
            <counter type="COMPLEXITY" missed="2" covered="3"/>
            <counter type="METHOD" missed="1" covered="3"/>
            <counter type="CLASS" missed="0" covered="1"/>
        </sourcefile>
        <counter type="INSTRUCTION" missed="3" covered="17"/>
        <counter type="BRANCH" missed="1" covered="1"/>
        <counter type="LINE" missed="2" covered="9"/>
        <counter type="COMPLEXITY" missed="2" covered="3"/>
        <counter type="METHOD" missed="1" covered="3"/>
        <counter type="CLASS" missed="0" covered="1"/>
    </package>
    <counter type="INSTRUCTION" missed="3" covered="17"/>
    <counter type="BRANCH" missed="1" covered="1"/>
    <counter type="LINE" missed="2" covered="9"/>
    <counter type="COMPLEXITY" missed="2" covered="3"/>
    <counter type="METHOD" missed="1" covered="3"/>
    <counter type="CLASS" missed="0" covered="1"/>
</report>
"""

# POM with JaCoCo already configured
POM_WITH_JACOCO = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>

    <build>
        <plugins>
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.11</version>
            </plugin>
        </plugins>
    </build>
</project>
"""

# POM without JaCoCo
POM_WITHOUT_JACOCO = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
            </plugin>
        </plugins>
    </build>
</project>
"""

# POM without build section
POM_MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>minimal-app</artifactId>
    <version>1.0.0</version>
</project>
"""

# POM without namespace
POM_NO_NAMESPACE = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>no-ns-app</artifactId>
    <version>1.0.0</version>
</project>
"""


class TestJacocoCoverageUtils:
    """Tests for JaCoCo XML parsing."""

    def test_load_from_jacoco_xml_basic(self, tmp_path: Path) -> None:
        """Test loading coverage data from a JaCoCo XML report."""
        # Create JaCoCo XML file
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)

        # Create source file path
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        # Parse coverage
        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # Verify coverage was parsed
        assert coverage_data is not None
        assert coverage_data.status == CoverageStatus.PARSED_SUCCESSFULLY
        assert coverage_data.function_name == "add"

    def test_load_from_jacoco_xml_covered_method(self, tmp_path: Path) -> None:
        """Test parsing a fully covered method."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)

        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # add method should be 100% covered (line 40-41 both covered)
        assert coverage_data.coverage == 100.0
        assert len(coverage_data.main_func_coverage.executed_lines) == 2
        assert len(coverage_data.main_func_coverage.unexecuted_lines) == 0

    def test_load_from_jacoco_xml_uncovered_method(self, tmp_path: Path) -> None:
        """Test parsing a fully uncovered method."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)

        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="subtract",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # subtract method should be 0% covered
        assert coverage_data.coverage == 0.0
        assert len(coverage_data.main_func_coverage.executed_lines) == 0
        assert len(coverage_data.main_func_coverage.unexecuted_lines) == 2

    def test_load_from_jacoco_xml_branch_coverage(self, tmp_path: Path) -> None:
        """Test parsing branch coverage data."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)

        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="multiply",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # multiply method should have branch coverage
        assert coverage_data.status == CoverageStatus.PARSED_SUCCESSFULLY
        # Line 60 has mb="1" cb="1" meaning 1 covered branch and 1 missed branch
        assert len(coverage_data.main_func_coverage.executed_branches) > 0
        assert len(coverage_data.main_func_coverage.unexecuted_branches) > 0

    def test_load_from_jacoco_xml_missing_file(self, tmp_path: Path) -> None:
        """Test handling of missing JaCoCo XML file."""
        # Non-existent file
        jacoco_xml = tmp_path / "nonexistent.xml"

        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # Should return empty coverage
        assert coverage_data.status == CoverageStatus.NOT_FOUND
        assert coverage_data.coverage == 0.0

    def test_load_from_jacoco_xml_invalid_xml(self, tmp_path: Path) -> None:
        """Test handling of invalid XML."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text("this is not valid xml")

        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # Should return empty coverage
        assert coverage_data.status == CoverageStatus.NOT_FOUND
        assert coverage_data.coverage == 0.0

    def test_load_from_jacoco_xml_no_matching_source(self, tmp_path: Path) -> None:
        """Test handling when source file is not found in report."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)

        # Source file that doesn't match
        source_path = tmp_path / "OtherClass.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(),
            source_code_path=source_path,
        )

        # Should return empty coverage (no matching sourcefile)
        assert coverage_data.status == CoverageStatus.NOT_FOUND
        assert coverage_data.coverage == 0.0

    def test_no_helper_functions_no_dependent_coverage(self, tmp_path: Path) -> None:
        """With zero helper functions, dependent_func_coverage stays None and total == main."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(helper_functions=[]),
            source_code_path=source_path,
        )

        assert coverage_data.dependent_func_coverage is None
        assert coverage_data.functions_being_tested == ["add"]
        assert coverage_data.coverage == 100.0  # add is fully covered

    def test_multiple_helpers_no_dependent_coverage(self, tmp_path: Path) -> None:
        """With more than one helper, dependent_func_coverage stays None (mirrors Python behavior)."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        helpers = [
            make_function_source("subtract", "Calculator.subtract", source_path),
            make_function_source("multiply", "Calculator.multiply", source_path),
        ]
        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(helper_functions=helpers),
            source_code_path=source_path,
        )

        assert coverage_data.dependent_func_coverage is None
        assert coverage_data.functions_being_tested == ["add"]

    def test_single_helper_found_in_jacoco_xml(self, tmp_path: Path) -> None:
        """With exactly one helper present in the JaCoCo XML, dependent_func_coverage is populated."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        # "add" is the main function; "multiply" is the helper
        helpers = [make_function_source("multiply", "Calculator.multiply", source_path)]
        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(helper_functions=helpers),
            source_code_path=source_path,
        )

        assert coverage_data.dependent_func_coverage is not None
        assert coverage_data.dependent_func_coverage.name == "Calculator.multiply"
        # multiply has LINE counter: missed=0, covered=3 → 100%
        assert coverage_data.dependent_func_coverage.coverage == 100.0
        assert coverage_data.functions_being_tested == ["add", "Calculator.multiply"]
        assert "Calculator.multiply" in coverage_data.graph

    def test_single_helper_absent_from_jacoco_xml(self, tmp_path: Path) -> None:
        """Helper listed in code_context but not in the JaCoCo XML → dependent_func_coverage stays None."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        helpers = [make_function_source("nonExistentMethod", "Calculator.nonExistentMethod", source_path)]
        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(helper_functions=helpers),
            source_code_path=source_path,
        )

        assert coverage_data.dependent_func_coverage is None
        assert coverage_data.functions_being_tested == ["add"]

    def test_total_coverage_aggregates_main_and_helper(self, tmp_path: Path) -> None:
        """Total coverage is computed over main + helper lines combined, not just main."""
        jacoco_xml = tmp_path / "jacoco.xml"
        jacoco_xml.write_text(SAMPLE_JACOCO_XML)
        source_path = tmp_path / "Calculator.java"
        source_path.write_text("// placeholder")

        # add (100% covered, lines 40-41) + subtract (0% covered, lines 50-51)
        # Combined: 2 executed + 2 unexecuted = 50% total
        helpers = [make_function_source("subtract", "Calculator.subtract", source_path)]
        coverage_data = JacocoCoverageUtils.load_from_jacoco_xml(
            jacoco_xml_path=jacoco_xml,
            function_name="add",
            code_context=create_mock_code_context(helper_functions=helpers),
            source_code_path=source_path,
        )

        assert coverage_data.dependent_func_coverage is not None
        assert coverage_data.main_func_coverage.coverage == 100.0
        assert coverage_data.dependent_func_coverage.coverage == 0.0
        # 2 covered (add) + 0 covered (subtract) out of 4 total lines = 50%
        assert coverage_data.coverage == 50.0


class TestJacocoPluginDetection:
    """Tests for JaCoCo plugin detection in pom.xml."""

    def test_is_jacoco_configured_with_plugin(self, tmp_path: Path) -> None:
        """Test detecting JaCoCo when it's configured."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_WITH_JACOCO)

        assert is_jacoco_configured(pom_path) is True

    def test_is_jacoco_configured_without_plugin(self, tmp_path: Path) -> None:
        """Test detecting JaCoCo when it's not configured."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_WITHOUT_JACOCO)

        assert is_jacoco_configured(pom_path) is False

    def test_is_jacoco_configured_minimal_pom(self, tmp_path: Path) -> None:
        """Test detecting JaCoCo in minimal pom without build section."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_MINIMAL)

        assert is_jacoco_configured(pom_path) is False

    def test_is_jacoco_configured_missing_file(self, tmp_path: Path) -> None:
        """Test detection when pom.xml doesn't exist."""
        pom_path = tmp_path / "pom.xml"

        assert is_jacoco_configured(pom_path) is False


class TestJacocoPluginAddition:
    """Tests for adding JaCoCo plugin to pom.xml."""

    def test_add_jacoco_plugin_to_minimal_pom(self, tmp_path: Path) -> None:
        """Test adding JaCoCo to a minimal pom.xml."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_MINIMAL)

        # Add JaCoCo plugin
        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is True

        # Verify it's now configured
        assert is_jacoco_configured(pom_path) is True

        # Verify the content
        content = pom_path.read_text()
        assert "jacoco-maven-plugin" in content
        assert "org.jacoco" in content
        assert "prepare-agent" in content
        assert "report" in content

    def test_add_jacoco_plugin_to_pom_with_build(self, tmp_path: Path) -> None:
        """Test adding JaCoCo to pom.xml that has a build section."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_WITHOUT_JACOCO)

        # Add JaCoCo plugin
        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is True

        # Verify it's now configured
        assert is_jacoco_configured(pom_path) is True

    def test_add_jacoco_plugin_already_present(self, tmp_path: Path) -> None:
        """Test adding JaCoCo when it's already configured."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_WITH_JACOCO)

        # Try to add JaCoCo plugin
        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is True  # Should succeed (already present)

        # Verify it's still configured
        assert is_jacoco_configured(pom_path) is True

    def test_add_jacoco_plugin_no_namespace(self, tmp_path: Path) -> None:
        """Test adding JaCoCo to pom.xml without XML namespace."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text(POM_NO_NAMESPACE)

        # Add JaCoCo plugin
        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is True

        # Verify it's now configured
        assert is_jacoco_configured(pom_path) is True

    def test_add_jacoco_plugin_missing_file(self, tmp_path: Path) -> None:
        """Test adding JaCoCo when pom.xml doesn't exist."""
        pom_path = tmp_path / "pom.xml"

        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is False

    def test_add_jacoco_plugin_invalid_xml(self, tmp_path: Path) -> None:
        """Test adding JaCoCo to invalid pom.xml."""
        pom_path = tmp_path / "pom.xml"
        pom_path.write_text("this is not valid xml")

        result = add_jacoco_plugin_to_pom(pom_path)
        assert result is False


class TestJacocoXmlPath:
    """Tests for JaCoCo XML path resolution."""

    def test_get_jacoco_xml_path(self, tmp_path: Path) -> None:
        """Test getting the expected JaCoCo XML path."""
        path = get_jacoco_xml_path(tmp_path)

        assert path == tmp_path / "target" / "site" / "jacoco" / "jacoco.xml"

    def test_jacoco_plugin_version(self) -> None:
        """Test that JaCoCo version constant is defined."""
        assert JACOCO_PLUGIN_VERSION == "0.8.13"
