"""GPU/device framework detection and synchronization AST generation."""

from __future__ import annotations

import ast


def detect_frameworks_from_code(code: str) -> dict[str, str]:
    """Detect GPU/device frameworks (torch, tensorflow, jax) used in the code by analyzing imports.

    Returns:
        A dictionary mapping framework names to their import aliases.
        For example: {"torch": "th", "tensorflow": "tf", "jax": "jax"}

    """
    frameworks: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frameworks

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name == "torch":
                    # Use asname if available, otherwise use the module name
                    frameworks["torch"] = alias.asname if alias.asname else module_name
                elif module_name == "tensorflow":
                    frameworks["tensorflow"] = alias.asname if alias.asname else module_name
                elif module_name == "jax":
                    frameworks["jax"] = alias.asname if alias.asname else module_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module.split(".")[0]
            if module_name == "torch" and "torch" not in frameworks:
                frameworks["torch"] = module_name
            elif module_name == "tensorflow" and "tensorflow" not in frameworks:
                frameworks["tensorflow"] = module_name
            elif module_name == "jax" and "jax" not in frameworks:
                frameworks["jax"] = module_name

    return frameworks


def create_device_sync_precompute_statements(used_frameworks: dict[str, str] | None) -> list[ast.stmt]:
    """Create AST statements to pre-compute device sync conditions before profiling.

    This moves the conditional checks (like is_available(), hasattr(), etc.) outside
    the timing block to avoid their overhead affecting the measurements.

    Args:
        used_frameworks: Dict mapping framework names to their import aliases

    Returns:
        List of AST statements that pre-compute sync conditions into boolean variables

    """
    if not used_frameworks:
        return []

    precompute_statements: list[ast.stmt] = []

    # PyTorch: pre-compute whether to sync CUDA or MPS
    if "torch" in used_frameworks:
        torch_alias = used_frameworks["torch"]
        # _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
        precompute_statements.append(
            ast.Assign(
                targets=[ast.Name(id="_codeflash_should_sync_cuda", ctx=ast.Store())],
                value=ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="cuda", ctx=ast.Load()
                                ),
                                attr="is_available",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="cuda", ctx=ast.Load()
                                ),
                                attr="is_initialized",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                    ],
                ),
                lineno=1,
            )
        )
        # _codeflash_should_sync_mps = (not _codeflash_should_sync_cuda and
        #     hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and
        #     hasattr(torch.mps, 'synchronize'))
        precompute_statements.append(
            ast.Assign(
                targets=[ast.Name(id="_codeflash_should_sync_mps", ctx=ast.Store())],
                value=ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_codeflash_should_sync_cuda", ctx=ast.Load())),
                        ast.Call(
                            func=ast.Name(id="hasattr", ctx=ast.Load()),
                            args=[
                                ast.Attribute(
                                    value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="backends", ctx=ast.Load()
                                ),
                                ast.Constant(value="mps"),
                            ],
                            keywords=[],
                        ),
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="backends", ctx=ast.Load()
                                    ),
                                    attr="mps",
                                    ctx=ast.Load(),
                                ),
                                attr="is_available",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                        ast.Call(
                            func=ast.Name(id="hasattr", ctx=ast.Load()),
                            args=[
                                ast.Attribute(
                                    value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="mps", ctx=ast.Load()
                                ),
                                ast.Constant(value="synchronize"),
                            ],
                            keywords=[],
                        ),
                    ],
                ),
                lineno=1,
            )
        )

    # JAX: pre-compute whether jax.block_until_ready exists
    if "jax" in used_frameworks:
        jax_alias = used_frameworks["jax"]
        # _codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')
        precompute_statements.append(
            ast.Assign(
                targets=[ast.Name(id="_codeflash_should_sync_jax", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="hasattr", ctx=ast.Load()),
                    args=[ast.Name(id=jax_alias, ctx=ast.Load()), ast.Constant(value="block_until_ready")],
                    keywords=[],
                ),
                lineno=1,
            )
        )

    # TensorFlow: pre-compute whether tf.test.experimental.sync_devices exists
    if "tensorflow" in used_frameworks:
        tf_alias = used_frameworks["tensorflow"]
        # _codeflash_should_sync_tf = hasattr(tf.test.experimental, 'sync_devices')
        precompute_statements.append(
            ast.Assign(
                targets=[ast.Name(id="_codeflash_should_sync_tf", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="hasattr", ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=tf_alias, ctx=ast.Load()), attr="test", ctx=ast.Load()
                            ),
                            attr="experimental",
                            ctx=ast.Load(),
                        ),
                        ast.Constant(value="sync_devices"),
                    ],
                    keywords=[],
                ),
                lineno=1,
            )
        )

    return precompute_statements


def create_device_sync_statements(
    used_frameworks: dict[str, str] | None, for_return_value: bool = False
) -> list[ast.stmt]:
    """Create AST statements for device synchronization using pre-computed conditions.

    Args:
        used_frameworks: Dict mapping framework names to their import aliases
                        (e.g., {'torch': 'th', 'tensorflow': 'tf', 'jax': 'jax'})
        for_return_value: If True, creates sync for after function call (includes JAX block_until_ready)

    Returns:
        List of AST statements for device synchronization using pre-computed boolean variables

    """
    if not used_frameworks:
        return []

    sync_statements: list[ast.stmt] = []

    # PyTorch synchronization using pre-computed conditions
    if "torch" in used_frameworks:
        torch_alias = used_frameworks["torch"]
        # if _codeflash_should_sync_cuda:
        #     torch.cuda.synchronize()
        # elif _codeflash_should_sync_mps:
        #     torch.mps.synchronize()
        cuda_sync = ast.If(
            test=ast.Name(id="_codeflash_should_sync_cuda", ctx=ast.Load()),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="cuda", ctx=ast.Load()
                            ),
                            attr="synchronize",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    )
                )
            ],
            orelse=[
                ast.If(
                    test=ast.Name(id="_codeflash_should_sync_mps", ctx=ast.Load()),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id=torch_alias, ctx=ast.Load()), attr="mps", ctx=ast.Load()
                                    ),
                                    attr="synchronize",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            )
                        )
                    ],
                    orelse=[],
                )
            ],
        )
        sync_statements.append(cuda_sync)

    # JAX synchronization (only after function call, using block_until_ready on return value)
    if "jax" in used_frameworks and for_return_value:
        jax_alias = used_frameworks["jax"]
        # if _codeflash_should_sync_jax:
        #     jax.block_until_ready(return_value)
        jax_sync = ast.If(
            test=ast.Name(id="_codeflash_should_sync_jax", ctx=ast.Load()),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=jax_alias, ctx=ast.Load()), attr="block_until_ready", ctx=ast.Load()
                        ),
                        args=[ast.Name(id="return_value", ctx=ast.Load())],
                        keywords=[],
                    )
                )
            ],
            orelse=[],
        )
        sync_statements.append(jax_sync)

    # TensorFlow synchronization using pre-computed condition
    if "tensorflow" in used_frameworks:
        tf_alias = used_frameworks["tensorflow"]
        # if _codeflash_should_sync_tf:
        #     tf.test.experimental.sync_devices()
        tf_sync = ast.If(
            test=ast.Name(id="_codeflash_should_sync_tf", ctx=ast.Load()),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=tf_alias, ctx=ast.Load()), attr="test", ctx=ast.Load()
                                ),
                                attr="experimental",
                                ctx=ast.Load(),
                            ),
                            attr="sync_devices",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    )
                )
            ],
            orelse=[],
        )
        sync_statements.append(tf_sync)

    return sync_statements
