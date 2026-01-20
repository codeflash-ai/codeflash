# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for building codeflash CLI as a standalone binary.

Build with: pyinstaller codeflash.spec

This creates a single-file executable that includes:
- The codeflash Python package
- JavaScript runtime files for JS/TS support
- Workflow templates for GitHub Actions
- All required dependencies
"""

import sys
from pathlib import Path

block_cipher = None

# Get the source directory
src_dir = Path(SPECPATH)

# Data files to include
datas = [
    # JavaScript runtime files
    (
        str(src_dir / 'codeflash' / 'languages' / 'javascript' / 'runtime' / '*.js'),
        'codeflash/languages/javascript/runtime'
    ),
    # Workflow templates
    (
        str(src_dir / 'codeflash' / 'cli_cmds' / 'workflows' / '*.yaml'),
        'codeflash/cli_cmds/workflows'
    ),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Tree-sitter languages
    'tree_sitter',
    'tree_sitter_python',
    'tree_sitter_javascript',
    'tree_sitter_typescript',
    # CLI dependencies
    'click',
    'inquirer',
    'inquirer.render',
    'inquirer.render.console',
    'readchar',
    'rich',
    'tomlkit',
    'gitpython',
    'git',
    # Testing frameworks
    'pytest',
    'junitparser',
    # Code analysis
    'libcst',
    'jedi',
    # Serialization
    'dill',
    'pydantic',
    # HTTP/API
    'requests',
    'posthog',
    # Other
    'humanize',
    'sentry_sdk',
    'unidiff',
    'isort',
    'coverage',
    'line_profiler',
    'platformdirs',
    'filelock',
    # Encoding
    'encodings',
    'encodings.utf_8',
    'encodings.ascii',
    'encodings.latin_1',
]

# Collect package metadata for packages that use importlib.metadata
from PyInstaller.utils.hooks import copy_metadata

# Packages that require their metadata to be available at runtime
metadata_packages = [
    'readchar',
    'codeflash',
    'pytest',
    'inquirer',
    'rich',
    'click',
    'gitpython',
    'pydantic',
    'posthog',
    'sentry_sdk',
]

extra_datas = []
for pkg in metadata_packages:
    try:
        extra_datas += copy_metadata(pkg)
    except Exception:
        pass  # Package might not be installed

a = Analysis(
    [str(src_dir / 'codeflash' / 'main.py')],
    pathex=[str(src_dir)],
    binaries=[],
    datas=datas + extra_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test dependencies to reduce size
        'matplotlib',
        'PIL',
        'tkinter',
        '_tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='codeflash',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
