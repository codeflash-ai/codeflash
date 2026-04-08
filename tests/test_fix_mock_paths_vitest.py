"""Test fix_jest_mock_paths function with vitest mocks."""

from pathlib import Path

from codeflash.languages.javascript.instrument import fix_jest_mock_paths


def test_fix_vitest_mock_paths():
    """Test that vi.mock() paths are fixed correctly."""
    # Simulate source at src/agents/workspace.ts importing from ../routing/session-key
    # Test at test/test_workspace.test.ts should mock ../src/routing/session-key, not ../routing/session-key

    test_code = """
vi.mock('../routing/session-key', () => ({
  isSubagentSessionKey: vi.fn(),
  isCronSessionKey: vi.fn(),
}));

import { filterBootstrapFilesForSession } from '../src/agents/workspace.js';
    """

    # Create temp directories and files for testing
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        # Create directory structure
        src = project / "src"
        src_agents = src / "agents"
        src_routing = src / "routing"
        test_dir = project / "test"

        src_agents.mkdir(parents=True)
        src_routing.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # Create files
        source_file = src_agents / "workspace.ts"
        source_file.write_text("export function filterBootstrapFilesForSession() {}")

        routing_file = src_routing / "session-key.ts"
        routing_file.write_text("export function isSubagentSessionKey() {}")

        test_file = test_dir / "test_workspace.test.ts"
        test_file.write_text(test_code)

        # Fix the paths
        fixed = fix_jest_mock_paths(test_code, test_file, source_file, test_dir)

        # Should change ../routing/session-key to ../src/routing/session-key
        assert "../src/routing/session-key" in fixed, f"Expected path to be fixed, got: {fixed}"
        assert "../routing/session-key" not in fixed or "../src/routing/session-key" in fixed


def test_fix_jest_mock_paths_still_works():
    """Test that jest.mock() paths are still fixed correctly."""
    test_code = """
jest.mock('../routing/session-key', () => ({
  isSubagentSessionKey: jest.fn(),
}));
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)
        src = project / "src"
        src_agents = src / "agents"
        src_routing = src / "routing"
        test_dir = project / "test"

        src_agents.mkdir(parents=True)
        src_routing.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        source_file = src_agents / "workspace.ts"
        source_file.write_text("")

        routing_file = src_routing / "session-key.ts"
        routing_file.write_text("")

        test_file = test_dir / "test_workspace.test.ts"
        test_file.write_text(test_code)

        fixed = fix_jest_mock_paths(test_code, test_file, source_file, test_dir)

        assert "../src/routing/session-key" in fixed


if __name__ == "__main__":
    test_fix_vitest_mock_paths()
    test_fix_jest_mock_paths_still_works()
    print("All tests passed!")
