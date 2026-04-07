"""Tests for inject_dom_snapshot_calls in React behavioral verification."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.testgen import inject_dom_snapshot_calls


def test_injects_after_fireEvent():
    source = """\
import codeflash from 'codeflash';
const { container } = codeflash.captureRender('Comp', '1', render, Comp);
fireEvent.click(screen.getByText('Add'));
fireEvent.change(input, { target: { value: 'hi' } });
"""
    result = inject_dom_snapshot_calls(source)
    assert "codeflash.snapshotDOM('after_click_1');" in result
    assert "codeflash.snapshotDOM('after_change_1');" in result


def test_skips_perf_mode():
    source = """\
import codeflash from 'codeflash';
const result = await codeflash.captureRenderPerf('Comp', '1', render, Comp);
fireEvent.click(screen.getByText('Add'));
"""
    result = inject_dom_snapshot_calls(source)
    assert "snapshotDOM" not in result


def test_skips_without_captureRender():
    source = """\
import codeflash from 'codeflash';
fireEvent.click(screen.getByText('Add'));
"""
    result = inject_dom_snapshot_calls(source)
    assert "snapshotDOM" not in result


def test_preserves_indentation():
    source = """\
import codeflash from 'codeflash';
const { container } = codeflash.captureRender('Comp', '1', render, Comp);
    fireEvent.click(btn);
        fireEvent.change(input, { target: { value: 'x' } });
"""
    result = inject_dom_snapshot_calls(source)
    lines = result.split("\n")
    # Find the snapshot lines and check their indentation
    snapshot_lines = [l for l in lines if "snapshotDOM" in l]
    assert len(snapshot_lines) == 2
    assert snapshot_lines[0].startswith("    codeflash.snapshotDOM")
    assert snapshot_lines[1].startswith("        codeflash.snapshotDOM")


def test_no_semicolons():
    """Real-world projects (e.g. Zustand) don't use semicolons."""
    source = """\
const { container } = codeflash.captureRender('Comp', '1', render, Comp)
    fireEvent.click(screen.getByText('button'))
    fireEvent.click(screen.getByTestId('test-shallow'))
"""
    result = inject_dom_snapshot_calls(source)
    assert "after_click_1" in result
    assert "after_click_2" in result


def test_sequential_counter():
    source = """\
import codeflash from 'codeflash';
const { container } = codeflash.captureRender('Comp', '1', render, Comp);
fireEvent.click(btn);
fireEvent.click(btn);
fireEvent.click(btn);
"""
    result = inject_dom_snapshot_calls(source)
    assert "after_click_1" in result
    assert "after_click_2" in result
    assert "after_click_3" in result
