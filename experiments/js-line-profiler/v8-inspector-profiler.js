/**
 * V8 Inspector-based Profiler
 *
 * Uses the built-in V8 inspector protocol to collect CPU profiling data.
 * This is the same mechanism used by Chrome DevTools.
 */

const inspector = require('inspector');
const session = new inspector.Session();

let isSessionConnected = false;

/**
 * Start the profiler.
 */
async function startProfiling() {
    if (!isSessionConnected) {
        session.connect();
        isSessionConnected = true;
    }

    return new Promise((resolve, reject) => {
        session.post('Profiler.enable', (err) => {
            if (err) return reject(err);

            session.post('Profiler.setSamplingInterval', { interval: 100 }, (err) => {
                if (err) return reject(err);

                session.post('Profiler.start', (err) => {
                    if (err) return reject(err);
                    resolve();
                });
            });
        });
    });
}

/**
 * Stop the profiler and get the profile data.
 */
async function stopProfiling() {
    return new Promise((resolve, reject) => {
        session.post('Profiler.stop', (err, { profile }) => {
            if (err) return reject(err);
            resolve(profile);
        });
    });
}

/**
 * Parse the V8 profile to extract line-level timings.
 */
function parseProfile(profile) {
    const lineTimings = {};

    // Build a map of node IDs to their hit counts
    const nodeHits = {};
    for (const sample of profile.samples || []) {
        nodeHits[sample] = (nodeHits[sample] || 0) + 1;
    }

    // Process nodes to extract line information
    function processNode(node, parentHits = 0) {
        const { callFrame } = node;
        const filename = callFrame.url || callFrame.scriptId;
        const lineNumber = callFrame.lineNumber + 1; // V8 uses 0-indexed lines
        const functionName = callFrame.functionName || '(anonymous)';

        const hits = nodeHits[node.id] || 0;

        if (filename && lineNumber > 0) {
            if (!lineTimings[filename]) {
                lineTimings[filename] = {};
            }
            if (!lineTimings[filename][lineNumber]) {
                lineTimings[filename][lineNumber] = {
                    hits: 0,
                    functionName,
                    selfTime: 0
                };
            }
            lineTimings[filename][lineNumber].hits += hits;
        }

        // Process children
        if (node.children) {
            for (const childId of node.children) {
                const childNode = findNode(profile.nodes, childId);
                if (childNode) {
                    processNode(childNode, hits);
                }
            }
        }
    }

    function findNode(nodes, id) {
        return nodes.find(n => n.id === id);
    }

    // Start from the root
    if (profile.nodes && profile.nodes.length > 0) {
        processNode(profile.nodes[0]);
    }

    // Calculate percentages
    const totalSamples = profile.samples?.length || 1;
    for (const filename of Object.keys(lineTimings)) {
        for (const line of Object.keys(lineTimings[filename])) {
            const data = lineTimings[filename][line];
            data.percentage = (data.hits / totalSamples * 100).toFixed(2);
        }
    }

    return lineTimings;
}

/**
 * Alternative: Use precise CPU profiling with tick processor.
 */
async function startPreciseProfiling() {
    if (!isSessionConnected) {
        session.connect();
        isSessionConnected = true;
    }

    return new Promise((resolve, reject) => {
        session.post('Profiler.enable', (err) => {
            if (err) return reject(err);

            // Use microsecond precision
            session.post('Profiler.setSamplingInterval', { interval: 10 }, (err) => {
                if (err) return reject(err);

                // Enable precise coverage if available
                session.post('Profiler.startPreciseCoverage', {
                    callCount: true,
                    detailed: true
                }, (err) => {
                    // Ignore error if not supported
                    session.post('Profiler.start', (err) => {
                        if (err) return reject(err);
                        resolve();
                    });
                });
            });
        });
    });
}

/**
 * Stop precise profiling and get coverage data.
 */
async function stopPreciseProfiling() {
    return new Promise((resolve, reject) => {
        // Get precise coverage
        session.post('Profiler.takePreciseCoverage', (coverageErr, coverageResult) => {
            // Get regular profile
            session.post('Profiler.stop', (err, { profile }) => {
                if (err) return reject(err);
                resolve({
                    profile,
                    coverage: coverageResult?.result || []
                });
            });
        });
    });
}

/**
 * Parse coverage data for line-level information.
 */
function parseCoverage(coverage) {
    const lineTimings = {};

    for (const script of coverage) {
        const scriptId = script.scriptId;
        const url = script.url;

        for (const func of script.functions) {
            const funcName = func.functionName || '(anonymous)';

            for (const range of func.ranges) {
                const startLine = range.startOffset; // Note: these are byte offsets
                const endLine = range.endOffset;
                const count = range.count;

                if (!lineTimings[url]) {
                    lineTimings[url] = {};
                }
                // For simplicity, use offset as key (would need source map for lines)
                const key = `offset:${startLine}-${endLine}`;
                lineTimings[url][key] = {
                    functionName: funcName,
                    count,
                    startOffset: startLine,
                    endOffset: endLine
                };
            }
        }
    }

    return lineTimings;
}

/**
 * Disconnect the session.
 */
function disconnect() {
    if (isSessionConnected) {
        session.post('Profiler.disable', () => {});
        session.disconnect();
        isSessionConnected = false;
    }
}

module.exports = {
    startProfiling,
    stopProfiling,
    parseProfile,
    startPreciseProfiling,
    stopPreciseProfiling,
    parseCoverage,
    disconnect
};
