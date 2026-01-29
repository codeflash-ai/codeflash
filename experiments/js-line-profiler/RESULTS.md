# Node.js Line Profiler Experiment Results

## Executive Summary

**Recommendation: Use custom `process.hrtime.bigint()` instrumentation for line-level profiling in Codeflash.**

Despite the significant overhead (2000-7500%), the custom instrumentation approach:
1. Correctly identifies hot spots with 100% accuracy
2. Provides precise per-line timing data
3. Works reliably with V8's JIT (after ~1000 iteration warmup)
4. Can leverage existing tree-sitter infrastructure

---

## Approaches Tested

### 1. V8 Inspector Sampling Profiler

**How it works:** Uses V8's built-in CPU profiler via the inspector protocol. Samples the call stack at regular intervals.

**Results:**
- Total samples: 6,028
- Correctly identified `reverseString` as hottest (61.76% of samples)
- Correctly identified `bubbleSort` inner loop (4.66%)
- `fibonacci` appeared as 1.91%

**Pros:**
- Very low overhead (~1-5%)
- No code modification required
- Built into Node.js

**Cons:**
- Sampling-based: misses short operations
- Only function-level granularity (not line-level)
- Cannot distinguish individual lines within a function
- 10μs minimum sampling interval limits precision

**Verdict:** Useful for high-level hotspot detection, but **not suitable** for line-level profiling.

---

### 2. Custom `process.hrtime.bigint()` Instrumentation

**How it works:** Insert timing calls around each statement, accumulate timings, report per-line statistics.

**Results:**

| Function | Baseline | Instrumented | Overhead |
|----------|----------|--------------|----------|
| fibonacci(30) | 132ns | 10.02μs | +7,511% |
| reverseString | 8.66μs | 200μs | +2,209% |
| bubbleSort | 343ns | 18.68μs | +5,341% |

**Timer Characteristics:**
- Average timer overhead: ~962ns per call
- Minimum: 0ns (cached)
- Maximum: 4.35ms (occasional GC pause)

**JIT Warmup Effect:**
- First batch: 189ns/call
- After warmup (batch 2+): ~29ns/call
- JIT stabilizes within 2,000 iterations (85% speedup)

**Accuracy Verification:**

Tested with known expensive/cheap operations:
```
Expected: Line 5 (array alloc) most expensive
Actual:   Line 5 = 49.8% of time ✓

Expected: toString() > arithmetic
Actual:   Line 3 (toString) = 14.9%, Line 4 (arithmetic) = 13.6% ✓
```

**Line-Level Results for bubbleSort:**
```
Line  4 (inner loop):   28.1% of time, 44,000 calls
Line  5 (comparison):   21.6% of time, 36,000 calls
Line  6 (swap temp):    20.6% of time, 17,000 calls
Line  8 (swap assign):  12.0% of time, 17,000 calls
Line  7 (swap assign):   9.2% of time, 17,000 calls
```

**Pros:**
- Precise per-line timing
- Correctly identifies relative costs
- Works with any JavaScript code
- No external dependencies

**Cons:**
- High overhead (2000-7500%)
- Requires AST transformation
- Timer overhead dominates for very fast lines

**Verdict:** **Best approach** for detailed optimization analysis. Overhead is acceptable for profiling runs.

---

## Key Technical Findings

### 1. Timer Precision

`process.hrtime.bigint()` provides nanosecond precision but:
- Minimum measurable time: ~28-30ns (after JIT warmup)
- Timer call overhead: ~30-40ns best case, ~1μs average
- Occasional spikes to milliseconds (GC/kernel scheduling)

### 2. JIT Impact

V8's JIT significantly affects measurements:
- Cold code: ~190ns/call for fibonacci
- Warm code: ~29ns/call (6.5x faster)
- Stabilization: ~1,000-2,000 iterations
- **Recommendation:** Always warmup before measuring

### 3. Measurement Consistency

Coefficient of variation across runs: 83.38% (high variance)
- Caused by JIT warmup and GC pauses
- Mitigation: Multiple runs, discard outliers, focus on relative %

### 4. Relative vs Absolute Accuracy

**Relative accuracy is excellent:**
- Correctly ranks operations by cost
- Identifies hot spots accurately
- Percentage-based reporting is reliable

**Absolute accuracy is moderate:**
- Timer overhead inflates small operations
- Should not rely on absolute nanosecond values for fast lines
- Use call counts + relative % instead

---

## Implementation Recommendations for Codeflash

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JavaScript Line Profiler                  │
├─────────────────────────────────────────────────────────────┤
│  1. Parse with tree-sitter                                  │
│  2. Identify statement boundaries                           │
│  3. Insert timing instrumentation                           │
│  4. Warmup for 1,000+ iterations                            │
│  5. Measure for 5,000+ iterations                           │
│  6. Report: per-line %, call counts, hot spots              │
└─────────────────────────────────────────────────────────────┘
```

### Instrumentation Strategy

```javascript
// Before:
function example() {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += compute(i);
    }
    return sum;
}

// After:
function example() {
    let __t;

    __t = process.hrtime.bigint();
    let sum = 0;
    __profiler.record('example', 2, process.hrtime.bigint() - __t);

    __t = process.hrtime.bigint();
    for (let i = 0; i < n; i++) {
        __profiler.record('example', 3, process.hrtime.bigint() - __t);

        __t = process.hrtime.bigint();
        sum += compute(i);
        __profiler.record('example', 4, process.hrtime.bigint() - __t);

        __t = process.hrtime.bigint();
    }
    __profiler.record('example', 3, process.hrtime.bigint() - __t);

    __t = process.hrtime.bigint();
    const __ret = sum;
    __profiler.record('example', 6, process.hrtime.bigint() - __t);
    return __ret;
}
```

### Special Cases to Handle

1. **Return statements:** Store value, record time, then return
2. **Loops:** Time loop overhead separately from body
3. **Conditionals:** Time condition evaluation and each branch
4. **Try/catch:** Wrap carefully to preserve exception semantics
5. **Async/await:** Handle promise timing correctly

### Output Format

```json
{
  "function": "bubbleSort",
  "file": "sort.js",
  "lines": [
    {"line": 4, "percent": 28.1, "calls": 44000, "avgNs": 42},
    {"line": 5, "percent": 21.6, "calls": 36000, "avgNs": 40},
    {"line": 6, "percent": 20.6, "calls": 17000, "avgNs": 80}
  ],
  "hotSpots": [4, 5, 6]
}
```

---

## Comparison Summary

| Approach | Line Granularity | Accuracy | Overhead | Complexity |
|----------|------------------|----------|----------|------------|
| V8 Sampling | Function only | Moderate | ~1-5% | Low |
| Custom hrtime | Per-line | High | 2000-7500% | Medium |

**Winner: Custom hrtime instrumentation**

---

## Files in This Experiment

- `target-functions.js` - Test functions to profile
- `custom-line-profiler.js` - Custom instrumentation implementation
- `v8-inspector-profiler.js` - V8 inspector-based profiler
- `run-experiment.js` - Main experiment runner
- `experiment-results.json` - Detailed timing data
- `RESULTS.md` - This summary document
