/**
 * Codeflash Babel Tracer Plugin
 *
 * A Babel plugin that instruments JavaScript/TypeScript functions for tracing.
 * This plugin wraps functions with tracing calls to capture:
 * - Function arguments
 * - Return values
 * - Execution time
 *
 * The plugin transforms:
 *   function foo(a, b) { return a + b; }
 *
 * Into:
 *   const __codeflash_tracer__ = require('codeflash/tracer');
 *   function foo(a, b) {
 *     return __codeflash_tracer__.wrap(function foo(a, b) { return a + b; }, 'foo', '/path/file.js', 1)
 *       .apply(this, arguments);
 *   }
 *
 * Supported function types:
 * - FunctionDeclaration: function foo() {}
 * - FunctionExpression: const foo = function() {}
 * - ArrowFunctionExpression: const foo = () => {}
 * - ClassMethod: class Foo { bar() {} }
 * - ObjectMethod: const obj = { foo() {} }
 *
 * Configuration (via plugin options or environment variables):
 * - functions: Array of function names to trace (traces all if not set)
 * - files: Array of file patterns to trace (traces all if not set)
 * - exclude: Array of patterns to exclude from tracing
 *
 * Usage with @babel/register:
 *   require('@babel/register')({
 *     plugins: [['codeflash/babel-tracer-plugin', { functions: ['myFunc'] }]],
 *   });
 *
 * Environment Variables:
 *   CODEFLASH_FUNCTIONS - JSON array of functions to trace
 *   CODEFLASH_TRACE_FILES - JSON array of file patterns to trace
 *   CODEFLASH_TRACE_EXCLUDE - JSON array of patterns to exclude
 */

'use strict';

const path = require('path');

// Parse environment variables for configuration
function getEnvConfig() {
    const config = {
        functions: null,
        files: null,
        exclude: null,
    };

    try {
        if (process.env.CODEFLASH_FUNCTIONS) {
            config.functions = JSON.parse(process.env.CODEFLASH_FUNCTIONS);
        }
    } catch (e) {
        console.error('[codeflash-babel] Failed to parse CODEFLASH_FUNCTIONS:', e.message);
    }

    try {
        if (process.env.CODEFLASH_TRACE_FILES) {
            config.files = JSON.parse(process.env.CODEFLASH_TRACE_FILES);
        }
    } catch (e) {
        console.error('[codeflash-babel] Failed to parse CODEFLASH_TRACE_FILES:', e.message);
    }

    try {
        if (process.env.CODEFLASH_TRACE_EXCLUDE) {
            config.exclude = JSON.parse(process.env.CODEFLASH_TRACE_EXCLUDE);
        }
    } catch (e) {
        console.error('[codeflash-babel] Failed to parse CODEFLASH_TRACE_EXCLUDE:', e.message);
    }

    return config;
}

/**
 * Check if a function should be traced based on configuration.
 *
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {string|null} className - Class name (for methods)
 * @param {Object} config - Plugin configuration
 * @returns {boolean} - True if function should be traced
 */
function shouldTraceFunction(funcName, fileName, className, config) {
    // Check exclude patterns first
    if (config.exclude && config.exclude.length > 0) {
        for (const pattern of config.exclude) {
            if (typeof pattern === 'string') {
                if (funcName === pattern || fileName.includes(pattern)) {
                    return false;
                }
            } else if (pattern instanceof RegExp) {
                if (pattern.test(funcName) || pattern.test(fileName)) {
                    return false;
                }
            }
        }
    }

    // Check file patterns
    if (config.files && config.files.length > 0) {
        const matchesFile = config.files.some(pattern => {
            if (typeof pattern === 'string') {
                return fileName.includes(pattern);
            }
            if (pattern instanceof RegExp) {
                return pattern.test(fileName);
            }
            return false;
        });
        if (!matchesFile) return false;
    }

    // Check function names
    if (config.functions && config.functions.length > 0) {
        const matchesName = config.functions.some(f => {
            if (typeof f === 'string') {
                return f === funcName || f === `${className}.${funcName}`;
            }
            // Support object format: { function: 'name', file: 'path', class: 'className' }
            if (typeof f === 'object' && f !== null) {
                if (f.function && f.function !== funcName) return false;
                if (f.file && !fileName.includes(f.file)) return false;
                if (f.class && f.class !== className) return false;
                return true;
            }
            return false;
        });
        if (!matchesName) return false;
    }

    return true;
}

/**
 * Check if a path should be excluded from tracing (node_modules, etc.)
 *
 * @param {string} fileName - File path
 * @returns {boolean} - True if file should be excluded
 */
function isExcludedPath(fileName) {
    // Always exclude node_modules
    if (fileName.includes('node_modules')) return true;

    // Exclude common test runner internals
    if (fileName.includes('jest-runner') || fileName.includes('jest-jasmine')) return true;
    if (fileName.includes('@vitest')) return true;

    // Exclude this plugin itself
    if (fileName.includes('codeflash/runtime')) return true;
    if (fileName.includes('babel-tracer-plugin')) return true;

    return false;
}

/**
 * Create the Babel plugin.
 *
 * @param {Object} babel - Babel object with types (t)
 * @returns {Object} - Babel plugin configuration
 */
module.exports = function codeflashTracerPlugin(babel) {
    const { types: t } = babel;

    // Merge environment config with plugin options
    const envConfig = getEnvConfig();

    return {
        name: 'codeflash-tracer',

        visitor: {
            Program: {
                enter(programPath, state) {
                    // Merge options from plugin config and environment
                    state.codeflashConfig = {
                        ...envConfig,
                        ...(state.opts || {}),
                    };

                    // Track whether we've added the tracer import
                    state.tracerImportAdded = false;

                    // Get file info
                    state.fileName = state.filename || state.file.opts.filename || 'unknown';

                    // Check if entire file should be excluded
                    if (isExcludedPath(state.fileName)) {
                        state.skipFile = true;
                        return;
                    }

                    state.skipFile = false;
                },

                exit(programPath, state) {
                    // Add tracer import if we instrumented any functions
                    if (state.tracerImportAdded) {
                        const tracerRequire = t.variableDeclaration('const', [
                            t.variableDeclarator(
                                t.identifier('__codeflash_tracer__'),
                                t.callExpression(
                                    t.identifier('require'),
                                    [t.stringLiteral('codeflash/tracer')]
                                )
                            ),
                        ]);

                        // Add at the beginning of the program
                        programPath.unshiftContainer('body', tracerRequire);
                    }
                },
            },

            // Handle: function foo() {}
            FunctionDeclaration(path, state) {
                if (state.skipFile) return;
                if (!path.node.id) return; // Skip anonymous functions

                const funcName = path.node.id.name;
                const lineNumber = path.node.loc ? path.node.loc.start.line : 0;

                if (!shouldTraceFunction(funcName, state.fileName, null, state.codeflashConfig)) {
                    return;
                }

                // Transform the function body to wrap with tracing
                wrapFunctionBody(t, path, funcName, state.fileName, lineNumber, null);
                state.tracerImportAdded = true;
            },

            // Handle: const foo = function() {} or const foo = () => {}
            VariableDeclarator(path, state) {
                if (state.skipFile) return;
                if (!t.isIdentifier(path.node.id)) return;
                if (!path.node.init) return;

                const init = path.node.init;
                if (!t.isFunctionExpression(init) && !t.isArrowFunctionExpression(init)) {
                    return;
                }

                const funcName = path.node.id.name;
                const lineNumber = path.node.loc ? path.node.loc.start.line : 0;

                if (!shouldTraceFunction(funcName, state.fileName, null, state.codeflashConfig)) {
                    return;
                }

                // Wrap the function expression with tracer.wrap()
                path.node.init = createWrapperCall(t, init, funcName, state.fileName, lineNumber, null);
                state.tracerImportAdded = true;
            },

            // Handle: class Foo { bar() {} }
            ClassMethod(path, state) {
                if (state.skipFile) return;
                if (path.node.kind === 'constructor') return; // Skip constructors for now

                const funcName = path.node.key.name || (path.node.key.value && String(path.node.key.value));
                if (!funcName) return;

                // Get class name from parent
                const classPath = path.findParent(p => t.isClassDeclaration(p) || t.isClassExpression(p));
                const className = classPath && classPath.node.id ? classPath.node.id.name : null;

                const lineNumber = path.node.loc ? path.node.loc.start.line : 0;

                if (!shouldTraceFunction(funcName, state.fileName, className, state.codeflashConfig)) {
                    return;
                }

                // Wrap the method body
                wrapMethodBody(t, path, funcName, state.fileName, lineNumber, className);
                state.tracerImportAdded = true;
            },

            // Handle: const obj = { foo() {} }
            ObjectMethod(path, state) {
                if (state.skipFile) return;

                const funcName = path.node.key.name || (path.node.key.value && String(path.node.key.value));
                if (!funcName) return;

                const lineNumber = path.node.loc ? path.node.loc.start.line : 0;

                if (!shouldTraceFunction(funcName, state.fileName, null, state.codeflashConfig)) {
                    return;
                }

                // Wrap the method body
                wrapMethodBody(t, path, funcName, state.fileName, lineNumber, null);
                state.tracerImportAdded = true;
            },
        },
    };
};

/**
 * Create a __codeflash_tracer__.wrap() call expression.
 *
 * @param {Object} t - Babel types
 * @param {Object} funcNode - The function AST node
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name
 * @returns {Object} - Call expression AST node
 */
function createWrapperCall(t, funcNode, funcName, fileName, lineNumber, className) {
    const args = [
        funcNode,
        t.stringLiteral(funcName),
        t.stringLiteral(fileName),
        t.numericLiteral(lineNumber),
    ];

    if (className) {
        args.push(t.stringLiteral(className));
    } else {
        args.push(t.nullLiteral());
    }

    return t.callExpression(
        t.memberExpression(
            t.identifier('__codeflash_tracer__'),
            t.identifier('wrap')
        ),
        args
    );
}

/**
 * Wrap a function declaration's body with tracing.
 * Transforms:
 *   function foo(a, b) { return a + b; }
 * Into:
 *   function foo(a, b) {
 *     const __original__ = function(a, b) { return a + b; };
 *     return __codeflash_tracer__.wrap(__original__, 'foo', 'file.js', 1, null).apply(this, arguments);
 *   }
 *
 * @param {Object} t - Babel types
 * @param {Object} path - Babel path
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name
 */
function wrapFunctionBody(t, path, funcName, fileName, lineNumber, className) {
    const node = path.node;
    const isAsync = node.async;
    const isGenerator = node.generator;

    // Create a copy of the original function as an expression
    const originalFunc = t.functionExpression(
        null, // anonymous
        node.params,
        node.body,
        isGenerator,
        isAsync
    );

    // Create the wrapper call
    const wrapperCall = createWrapperCall(t, originalFunc, funcName, fileName, lineNumber, className);

    // Create: return __codeflash_tracer__.wrap(...).apply(this, arguments)
    const applyCall = t.callExpression(
        t.memberExpression(wrapperCall, t.identifier('apply')),
        [t.thisExpression(), t.identifier('arguments')]
    );

    const returnStatement = t.returnStatement(applyCall);

    // Replace the function body
    node.body = t.blockStatement([returnStatement]);
}

/**
 * Wrap a method's body with tracing.
 * Similar to wrapFunctionBody but preserves method semantics.
 *
 * @param {Object} t - Babel types
 * @param {Object} path - Babel path
 * @param {string} funcName - Function name
 * @param {string} fileName - File path
 * @param {number} lineNumber - Line number
 * @param {string|null} className - Class name
 */
function wrapMethodBody(t, path, funcName, fileName, lineNumber, className) {
    const node = path.node;
    const isAsync = node.async;
    const isGenerator = node.generator;

    // Create a copy of the original function as an expression
    const originalFunc = t.functionExpression(
        null, // anonymous
        node.params,
        node.body,
        isGenerator,
        isAsync
    );

    // Create the wrapper call
    const wrapperCall = createWrapperCall(t, originalFunc, funcName, fileName, lineNumber, className);

    // Create: return __codeflash_tracer__.wrap(...).apply(this, arguments)
    const applyCall = t.callExpression(
        t.memberExpression(wrapperCall, t.identifier('apply')),
        [t.thisExpression(), t.identifier('arguments')]
    );

    let returnStatement;
    if (isAsync) {
        // For async methods, we need to await the result
        returnStatement = t.returnStatement(t.awaitExpression(applyCall));
    } else {
        returnStatement = t.returnStatement(applyCall);
    }

    // Replace the function body
    node.body = t.blockStatement([returnStatement]);
}

// Export helper functions for testing
module.exports.shouldTraceFunction = shouldTraceFunction;
module.exports.isExcludedPath = isExcludedPath;
module.exports.getEnvConfig = getEnvConfig;
