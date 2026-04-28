# Jest

Jest is a comprehensive JavaScript testing framework designed for delightful testing experiences. It provides a complete testing solution that works out of the box for most JavaScript projects, featuring instant feedback through intelligent watch mode, powerful snapshot testing, extensive mocking capabilities, built-in code coverage reporting, and seamless integration with modern JavaScript tooling including Babel, TypeScript, webpack, and various bundlers.

## Package Information

- **Package Name**: jest
- **Package Type**: npm
- **Language**: TypeScript/JavaScript
- **Installation**: `npm install jest`

## Core Imports

```typescript
import { runCLI, createTestScheduler, SearchSource, getVersion, run, buildArgv } from "jest";
import type { Config } from "jest";
```

For CommonJS:

```javascript
const { runCLI, createTestScheduler, SearchSource, getVersion, run, buildArgv } = require("jest");
```

## Basic Usage

Jest can be used programmatically or via CLI:

```typescript
import { runCLI } from "jest";

// Run Jest programmatically
const { results, globalConfig } = await runCLI(
  { roots: ["<rootDir>/src"], testMatch: ["**/__tests__/**/*.test.js"] },
  ["./src"]
);

console.log(`Tests completed: ${results.numTotalTests}`);
console.log(`Tests passed: ${results.numPassedTests}`);
```

CLI usage:

```bash
# Run all tests
jest

# Run tests in watch mode
jest --watch

# Run with coverage
jest --coverage

# Run specific test files
jest user.test.js
```

## Architecture

Jest is built around several key components:

- **Test Runner**: Core engine that discovers, schedules, and executes tests
- **CLI Interface**: Command-line interface for running tests with extensive options
- **SearchSource**: Test file discovery and filtering system
- **TestScheduler**: Test execution scheduling and reporter management
- **Configuration System**: Flexible configuration for projects and global settings
- **Reporter System**: Extensible test result reporting and output formatting

## Capabilities

### Test Running and Execution

Core test running functionality including programmatic API and CLI runner with comprehensive test discovery and execution capabilities.

```typescript { .api }
function runCLI(
  argv: Config.Argv,
  projects: Array<string>
): Promise<{
  results: AggregatedResult;
  globalConfig: Config.GlobalConfig;
}>;

function run(maybeArgv?: Array<string>, project?: string): Promise<void>;
```

[Test Runner](./test-runner.md)

### CLI Usage and Options

Command-line interface providing over 70 options for test execution, coverage collection, watch mode, output formatting, and project configuration.

```typescript { .api }
function buildArgv(maybeArgv?: Array<string>): Promise<Config.Argv>;
```

[CLI Usage](./cli-usage.md)

### Test Discovery and Search

Advanced test file discovery system with pattern matching, dependency tracking, and change detection for optimized test runs.

```typescript { .api }
class SearchSource {
  constructor(context: TestContext);
  isTestFilePath(path: string): boolean;
  findMatchingTests(testPathPatternsExecutor: TestPathPatternsExecutor): SearchResult;
  findTestsByPaths(paths: Array<string>): SearchResult;
  findRelatedTests(allPaths: Set<string>, collectCoverage: boolean): Promise<SearchResult>;
}
```

[Test Discovery](./test-discovery.md)

### Configuration Management

Comprehensive configuration system supporting global settings, per-project configuration, and extensive customization options for all aspects of test execution.

```typescript { .api }
interface Config {
  // Main configuration interface (alias for Config.InitialOptions)
  testEnvironment?: string;
  testMatch?: Array<string>;
  transform?: Record<string, string>;
  setupFilesAfterEnv?: Array<string>;
  moduleNameMapping?: Record<string, string>;
  collectCoverage?: boolean;
}

interface Config.GlobalConfig {
  bail: number;
  collectCoverage: boolean;
  maxWorkers: number;
  rootDir: string;
  watch: boolean;
  watchAll: boolean;
}
```

[Configuration](./configuration.md)

### Test Scheduling and Reporting

Test execution scheduling with multi-process coordination, reporter management, and comprehensive result aggregation.

```typescript { .api }
function createTestScheduler(
  globalConfig: Config.GlobalConfig,
  context: TestSchedulerContext
): Promise<TestScheduler>;

class TestScheduler {
  constructor(globalConfig: Config.GlobalConfig, context: TestSchedulerContext);
  addReporter(reporter: Reporter): void;
  scheduleTests(tests: Array<Test>, watcher: TestWatcher): Promise<AggregatedResult>;
}
```

[Test Scheduling](./test-scheduling.md)

## Utility Functions

```typescript { .api }
function getVersion(): string;
```

Returns the current Jest version.

## Core Types

```typescript { .api }
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

interface SearchResult {
  noSCM?: boolean;
  stats?: Stats;
  collectCoverageFrom?: Set<string>;
  tests: Array<Test>;
  total?: number;
}

interface TestContext {
  config: Config.ProjectConfig;
  hasteFS: IHasteFS;
  moduleMap: IModuleMap;
  resolver: IResolver;
}

interface TestSchedulerContext extends ReporterContext, TestRunnerContext {}

interface Test {
  context: TestContext;
  duration?: number;
  path: string;
}
```