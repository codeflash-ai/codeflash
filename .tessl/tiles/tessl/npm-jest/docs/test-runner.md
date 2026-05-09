# Test Runner

Jest's test runner provides both programmatic and CLI-based test execution with comprehensive configuration options, result aggregation, and performance optimization through parallel execution.

## Capabilities

### Programmatic Test Execution

Run Jest tests programmatically with full control over configuration and execution parameters.

```typescript { .api }
/**
 * Runs Jest CLI programmatically with specified configuration and projects
 * @param argv - Command line arguments and configuration
 * @param projects - Array of project paths to run tests on
 * @returns Promise resolving to test results and global configuration
 */
function runCLI(
  argv: Config.Argv,
  projects: Array<string>
): Promise<{
  results: AggregatedResult;
  globalConfig: Config.GlobalConfig;
}>;

interface AggregatedResult {
  numTotalTests: number;
  numPassedTests: number;
  numFailedTests: number;
  numPendingTests: number;
  numRuntimeErrorTestSuites: number;
  numTotalTestSuites: number;
  numPassedTestSuites: number;
  numFailedTestSuites: number;
  numPendingTestSuites: number;
  openHandles: Array<Error>;
  snapshot: SnapshotSummary;
  success: boolean;
  startTime: number;
  testResults: Array<TestResult>;
  wasInterrupted: boolean;
}
```

**Usage Examples:**

```typescript
import { runCLI, buildArgv } from "jest";

// Basic test run
async function runTests() {
  const argv = await buildArgv(["--testMatch=**/*.test.js"]);
  const { results, globalConfig } = await runCLI(argv, ["./src"]);
  
  console.log(`${results.numPassedTests}/${results.numTotalTests} tests passed`);
  
  if (!results.success) {
    console.error("Tests failed!");
    process.exit(1);
  }
}

// Run with coverage and JSON output
async function runTestsWithCoverage() {
  const argv = await buildArgv([
    "--coverage",
    "--json",
    "--outputFile=test-results.json"
  ]);
  
  const { results } = await runCLI(argv, [process.cwd()]);
  return results;
}

// Run specific test files
async function runSpecificTests(testFiles: string[]) {
  const argv = await buildArgv([
    "--runTestsByPath",
    ...testFiles
  ]);
  
  const { results } = await runCLI(argv, [process.cwd()]);
  return results;
}
```

### CLI Entry Point

Main CLI runner that handles argument parsing and delegates to the programmatic API.

```typescript { .api }
/**
 * Main CLI entry point for Jest
 * @param maybeArgv - Optional command line arguments (defaults to process.argv.slice(2))
 * @param project - Optional project path
 * @returns Promise that resolves when Jest execution completes
 */
function run(maybeArgv?: Array<string>, project?: string): Promise<void>;
```

**Usage Example:**

```typescript
import { run } from "jest";

// Run Jest with default arguments
await run();

// Run Jest with custom arguments
await run(["--watch", "--testPathPatterns=src/components"]);

// Run Jest for specific project
await run(["--coverage"], "./my-project");
```

### Argument Parsing and Validation

Parse and validate command line arguments for Jest execution.

```typescript { .api }
/**
 * Builds and validates command line arguments for Jest
 * @param maybeArgv - Optional command line arguments
 * @returns Promise resolving to parsed and validated arguments
 */
function buildArgv(maybeArgv?: Array<string>): Promise<Config.Argv>;

interface Config.Argv {
  // Test execution options
  all?: boolean;
  bail?: boolean | number;
  findRelatedTests?: boolean;
  listTests?: boolean;
  onlyChanged?: boolean;
  onlyFailures?: boolean;
  passWithNoTests?: boolean;
  runInBand?: boolean;
  runTestsByPath?: boolean;
  testNamePattern?: string;
  testPathPatterns?: Array<string>;
  testTimeout?: number;
  
  // Watch mode options
  watch?: boolean;
  watchAll?: boolean;
  watchPathIgnorePatterns?: Array<string>;
  
  // Coverage options
  collectCoverage?: boolean;
  coverage?: boolean;
  collectCoverageFrom?: Array<string>;
  coverageDirectory?: string;
  coveragePathIgnorePatterns?: Array<string>;
  coverageProvider?: "babel" | "v8";
  coverageReporters?: Array<string>;
  coverageThreshold?: Record<string, number>;
  
  // Output options
  json?: boolean;
  outputFile?: string;
  verbose?: boolean;
  silent?: boolean;
  noStackTrace?: boolean;
  color?: boolean;
  colors?: boolean;
  
  // Configuration options
  config?: string;
  rootDir?: string;
  roots?: Array<string>;
  projects?: Array<string>;
  maxWorkers?: number | string;
  cache?: boolean;
  clearCache?: boolean;
  debug?: boolean;
  updateSnapshot?: boolean;
}
```

### Integration Patterns

Common patterns for integrating Jest into build tools and custom workflows:

**Build Tool Integration:**

```typescript
import { runCLI, buildArgv } from "jest";

async function buildToolIntegration(options: {
  testFiles?: string[];
  coverage?: boolean;
  watch?: boolean;
}) {
  const args = [];
  
  if (options.coverage) args.push("--coverage");
  if (options.watch) args.push("--watch");
  if (options.testFiles) {
    args.push("--runTestsByPath", ...options.testFiles);
  }
  
  const argv = await buildArgv(args);
  const { results } = await runCLI(argv, [process.cwd()]);
  
  return {
    success: results.success,
    testCount: results.numTotalTests,
    passedTests: results.numPassedTests,
    failedTests: results.numFailedTests,
    coverageMap: results.coverageMap
  };
}
```

**CI/CD Integration:**

```typescript
import { runCLI, buildArgv } from "jest";

async function runTestsInCI() {
  const argv = await buildArgv([
    "--ci",
    "--coverage",
    "--json",
    "--outputFile=test-results.json",
    "--coverageReporters=text-lcov",
    "--coverageDirectory=coverage"
  ]);
  
  try {
    const { results, globalConfig } = await runCLI(argv, [process.cwd()]);
    
    // Log summary
    console.log(`Tests: ${results.numPassedTests}/${results.numTotalTests} passed`);
    console.log(`Test Suites: ${results.numPassedTestSuites}/${results.numTotalTestSuites} passed`);
    
    if (!results.success) {
      console.error("❌ Tests failed");
      process.exit(1);
    }
    
    console.log("✅ All tests passed");
  } catch (error) {
    console.error("Error running tests:", error);
    process.exit(1);
  }
}
```

**Custom Test Discovery:**

```typescript
import { runCLI, buildArgv } from "jest";
import * as fs from "fs";
import * as path from "path";

async function runTestsForChangedFiles(changedFiles: string[]) {
  // Find test files related to changed source files
  const testFiles = changedFiles
    .filter(file => file.endsWith('.js') || file.endsWith('.ts'))
    .map(file => {
      const testFile = file.replace(/\.(js|ts)$/, '.test.$1');
      return fs.existsSync(testFile) ? testFile : null;
    })
    .filter(Boolean) as string[];
  
  if (testFiles.length === 0) {
    console.log("No test files found for changed files");
    return;
  }
  
  const argv = await buildArgv([
    "--runTestsByPath",
    ...testFiles
  ]);
  
  const { results } = await runCLI(argv, [process.cwd()]);
  return results;
}
```