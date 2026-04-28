# Test Discovery

Jest's test discovery system provides advanced capabilities for finding, filtering, and organizing test files with pattern matching, dependency tracking, and change detection for optimized test runs.

## Capabilities

### SearchSource Class

The SearchSource class is the core component responsible for test file discovery and filtering.

```typescript { .api }
/**
 * Core class for finding and filtering test files
 */
class SearchSource {
  constructor(context: TestContext);
  
  /**
   * Determines if a given path is a test file based on configuration
   * @param path - File path to check
   * @returns True if the path matches test file patterns
   */
  isTestFilePath(path: string): boolean;
  
  /**
   * Finds tests matching the given path patterns
   * @param testPathPatternsExecutor - Pattern executor for test paths
   * @returns Search results with matching tests
   */
  findMatchingTests(testPathPatternsExecutor: TestPathPatternsExecutor): SearchResult;
  
  /**
   * Finds tests by exact file paths
   * @param paths - Array of exact file paths
   * @returns Search results with specified test files
   */
  findTestsByPaths(paths: Array<string>): SearchResult;
  
  /**
   * Finds tests related to the given source file paths
   * @param allPaths - Set of source file paths
   * @param collectCoverage - Whether to collect coverage information
   * @returns Promise resolving to search results with related tests
   */
  findRelatedTests(allPaths: Set<string>, collectCoverage: boolean): Promise<SearchResult>;
  
  /**
   * Main method to get test paths based on configuration and options
   * @param globalConfig - Global Jest configuration
   * @param projectConfig - Project-specific configuration
   * @param changedFiles - Optional changed files information
   * @param filter - Optional filter function
   * @returns Promise resolving to comprehensive search results
   */
  getTestPaths(
    globalConfig: Config.GlobalConfig,
    projectConfig: Config.ProjectConfig,
    changedFiles?: ChangedFiles,
    filter?: Filter
  ): Promise<SearchResult>;
}

interface SearchResult {
  noSCM?: boolean;
  stats?: Stats;
  collectCoverageFrom?: Set<string>;
  tests: Array<Test>;
  total?: number;
}

interface Stats {
  roots: number;
  testMatch: number;
  testPathIgnorePatterns: number;
  testRegex: number;
  testPathPatterns?: number;
}
```

**Usage Examples:**

```typescript
import { SearchSource } from "jest";

// Create SearchSource instance
const searchSource = new SearchSource(testContext);

// Find all test files
const allTests = await searchSource.getTestPaths(
  globalConfig,
  projectConfig
);

console.log(`Found ${allTests.tests.length} test files`);

// Check if a file is a test file
const isTest = searchSource.isTestFilePath("src/__tests__/utils.test.js");
console.log(`Is test file: ${isTest}`);

// Find tests related to changed source files
const changedFiles = new Set(["src/utils.js", "src/components/Button.js"]);
const relatedTests = await searchSource.findRelatedTests(changedFiles, false);

console.log(`Found ${relatedTests.tests.length} related tests`);
```

### Test Pattern Matching

Advanced pattern matching for test file discovery:

```typescript
// Find tests by exact paths
const specificTests = searchSource.findTestsByPaths([
  "src/components/Button.test.js",
  "src/utils/helpers.test.js"
]);

// Find tests matching patterns (via getTestPaths)
const patternTests = await searchSource.getTestPaths(
  {
    ...globalConfig,
    testPathPatterns: ["components", "utils"]
  },
  projectConfig
);
```

### Change Detection Integration

Optimize test runs by finding tests related to changed files:

```typescript
import { SearchSource } from "jest";

async function runTestsForChangedFiles(
  searchSource: SearchSource,
  changedFiles: string[]
) {
  // Find tests related to changed source files
  const relatedTests = await searchSource.findRelatedTests(
    new Set(changedFiles),
    true // collectCoverage
  );
  
  if (relatedTests.tests.length === 0) {
    console.log("No tests found for changed files");
    return null;
  }
  
  return relatedTests;
}

// Usage with git integration
async function findTestsForGitChanges() {
  const changedFiles = await getChangedFilesFromGit();
  const relatedTests = await searchSource.findRelatedTests(
    new Set(changedFiles),
    false
  );
  
  return relatedTests.tests.map(test => test.path);
}
```

### Custom Test Discovery Patterns

Implement custom test discovery logic:

```typescript
import { SearchSource } from "jest";

class CustomTestDiscovery {
  constructor(private searchSource: SearchSource) {}
  
  async findTestsByFeature(featureName: string) {
    // Find all tests
    const allTests = await this.searchSource.getTestPaths(
      globalConfig,
      projectConfig
    );
    
    // Filter by feature directory or naming convention
    const featureTests = allTests.tests.filter(test =>
      test.path.includes(`features/${featureName}`) ||
      test.path.includes(`${featureName}.test.`)
    );
    
    return {
      ...allTests,
      tests: featureTests,
      total: featureTests.length
    };
  }
  
  async findTestsByTags(tags: string[]) {
    const allTests = await this.searchSource.getTestPaths(
      globalConfig,
      projectConfig
    );
    
    // Filter tests based on file content or naming patterns
    const taggedTests = allTests.tests.filter(test => {
      const filename = test.path.toLowerCase();
      return tags.some(tag => filename.includes(tag.toLowerCase()));
    });
    
    return {
      ...allTests,
      tests: taggedTests,
      total: taggedTests.length
    };
  }
  
  async findSlowTests(thresholdMs: number = 1000) {
    // This would typically require historical test timing data
    // Implementation would depend on custom test result storage
    const allTests = await this.searchSource.getTestPaths(
      globalConfig,
      projectConfig
    );
    
    // Example: identify tests by naming convention
    const potentiallySlowTests = allTests.tests.filter(test =>
      test.path.includes("integration") ||
      test.path.includes("e2e") ||
      test.path.includes("slow")
    );
    
    return {
      ...allTests,
      tests: potentiallySlowTests,
      total: potentiallySlowTests.length
    };
  }
}
```

### Performance Optimization

Optimize test discovery for large codebases:

```typescript
async function optimizedTestDiscovery(
  searchSource: SearchSource,
  options: {
    useCache?: boolean;
    maxFiles?: number;
    changedFilesOnly?: boolean;
  } = {}
) {
  if (options.changedFilesOnly) {
    // Only find tests related to changed files
    const changedFiles = await getChangedFiles();
    return searchSource.findRelatedTests(new Set(changedFiles), false);
  }
  
  // Get all tests with potential limits
  const allTests = await searchSource.getTestPaths(
    globalConfig,
    projectConfig
  );
  
  if (options.maxFiles && allTests.tests.length > options.maxFiles) {
    // Limit test count for performance
    const limitedTests = allTests.tests.slice(0, options.maxFiles);
    console.warn(`Limited to ${options.maxFiles} tests (found ${allTests.tests.length})`);
    
    return {
      ...allTests,
      tests: limitedTests,
      total: limitedTests.length
    };
  }
  
  return allTests;
}
```

## Integration with Test Execution

The SearchSource integrates seamlessly with Jest's test execution pipeline:

```typescript
import { SearchSource, createTestScheduler } from "jest";

async function discoverAndRunTests() {
  // 1. Discover tests
  const searchSource = new SearchSource(testContext);
  const searchResult = await searchSource.getTestPaths(
    globalConfig,
    projectConfig
  );
  
  // 2. Create scheduler
  const scheduler = await createTestScheduler(globalConfig, schedulerContext);
  
  // 3. Execute discovered tests
  const results = await scheduler.scheduleTests(
    searchResult.tests,
    testWatcher
  );
  
  return results;
}
```

Jest's test discovery system provides the foundation for intelligent test execution, enabling optimized test runs based on code changes, file patterns, and custom discovery logic.