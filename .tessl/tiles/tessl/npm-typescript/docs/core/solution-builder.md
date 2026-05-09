# Solution Builder

Solution Builder provides APIs for building TypeScript projects with project references, enabling efficient compilation of large codebases and monorepos through incremental builds and dependency management.

## Capabilities

### Solution Builder Creation

Create solution builders for building projects with project references.

```typescript { .api }
/**
 * Create a solution builder host for non-watch mode compilation
 * @param system - System interface for file operations
 * @param createProgram - Optional custom program creator
 * @param reportDiagnostic - Optional diagnostic reporter
 * @param reportSolutionBuilderStatus - Optional status reporter
 * @param reportErrorSummary - Optional error summary reporter
 * @returns Solution builder host instance
 */
function createSolutionBuilderHost<T extends BuilderProgram = EmitAndSemanticDiagnosticsBuilderProgram>(
  system?: System,
  createProgram?: CreateProgram<T>,
  reportDiagnostic?: DiagnosticReporter,
  reportSolutionBuilderStatus?: DiagnosticReporter,
  reportErrorSummary?: ReportEmitErrorSummary
): SolutionBuilderHost<T>;

/**
 * Create a solution builder host with file watching capabilities
 * @param system - System interface for file operations
 * @param createProgram - Optional custom program creator
 * @param reportDiagnostic - Optional diagnostic reporter
 * @param reportSolutionBuilderStatus - Optional status reporter
 * @param reportWatchStatus - Optional watch status reporter
 * @returns Solution builder host with watch capabilities
 */
function createSolutionBuilderWithWatchHost<T extends BuilderProgram = EmitAndSemanticDiagnosticsBuilderProgram>(
  system?: System,
  createProgram?: CreateProgram<T>,
  reportDiagnostic?: DiagnosticReporter,
  reportSolutionBuilderStatus?: DiagnosticReporter,
  reportWatchStatus?: WatchStatusReporter
): SolutionBuilderWithWatchHost<T>;

/**
 * Create a solution builder for building multiple projects
 * @param host - Solution builder host
 * @param rootNames - Array of project config file paths
 * @param defaultOptions - Default build options
 * @returns Solution builder instance
 */
function createSolutionBuilder<T extends BuilderProgram>(
  host: SolutionBuilderHost<T>,
  rootNames: readonly string[],
  defaultOptions: BuildOptions
): SolutionBuilder<T>;

/**
 * Create a solution builder with file watching
 * @param host - Solution builder host with watch capabilities
 * @param rootNames - Array of project config file paths
 * @param defaultOptions - Default build options
 * @param baseWatchOptions - Optional watch configuration
 * @returns Solution builder instance with watch mode
 */
function createSolutionBuilderWithWatch<T extends BuilderProgram>(
  host: SolutionBuilderWithWatchHost<T>,
  rootNames: readonly string[],
  defaultOptions: BuildOptions,
  baseWatchOptions?: WatchOptions
): SolutionBuilder<T>;

/**
 * Create a diagnostic reporter for builder status messages
 * @param system - System interface for output
 * @param pretty - Whether to use colored output
 * @returns Diagnostic reporter function
 */
function createBuilderStatusReporter(
  system: System,
  pretty?: boolean
): DiagnosticReporter;
```

**Usage Example:**

```typescript
import * as ts from 'typescript';

// Create diagnostic reporter (simple function implementing DiagnosticReporter type)
const diagnosticReporter: ts.DiagnosticReporter = (diagnostic) => {
  console.error(ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
};

// Create a solution builder host
const host = ts.createSolutionBuilderHost(
  ts.sys,
  undefined,
  diagnosticReporter,
  ts.createBuilderStatusReporter(ts.sys, true),
  (errorCount) => console.log(`Errors: ${errorCount}`)
);

// Create the solution builder
const builder = ts.createSolutionBuilder(
  host,
  ['./packages/app/tsconfig.json', './packages/lib/tsconfig.json'],
  { verbose: true }
);

// Build all projects
const exitStatus = builder.build();
```

### Solution Builder Interface

Main interface for building and cleaning projects with project references.

```typescript { .api }
interface SolutionBuilder<T extends BuilderProgram> {
  /**
   * Build a specific project or all projects
   * @param project - Optional project path to build (builds all if not specified)
   * @param cancellationToken - Optional cancellation token
   * @param writeFile - Optional custom file writer
   * @param getCustomTransformers - Optional custom transformer provider
   * @returns Exit status code
   */
  build(
    project?: string,
    cancellationToken?: CancellationToken,
    writeFile?: WriteFileCallback,
    getCustomTransformers?: (project: string) => CustomTransformers
  ): ExitStatus;

  /**
   * Clean build outputs for a project or all projects
   * @param project - Optional project path to clean (cleans all if not specified)
   * @returns Exit status code
   */
  clean(project?: string): ExitStatus;

  /**
   * Build only the project references of a specific project
   * @param project - Project path
   * @param cancellationToken - Optional cancellation token
   * @param writeFile - Optional custom file writer
   * @param getCustomTransformers - Optional custom transformer provider
   * @returns Exit status code
   */
  buildReferences(
    project: string,
    cancellationToken?: CancellationToken,
    writeFile?: WriteFileCallback,
    getCustomTransformers?: (project: string) => CustomTransformers
  ): ExitStatus;

  /**
   * Clean build outputs for project references
   * @param project - Optional project path
   * @returns Exit status code
   */
  cleanReferences(project?: string): ExitStatus;

  /**
   * Get the next invalidated project that needs to be built
   * @param cancellationToken - Optional cancellation token
   * @returns Next invalidated project or undefined if all up-to-date
   */
  getNextInvalidatedProject(
    cancellationToken?: CancellationToken
  ): InvalidatedProject<T> | undefined;
}
```

### Build Options

Configuration options for solution builder builds.

```typescript { .api }
interface BuildOptions {
  /** Perform a dry run without writing files */
  dry?: boolean;
  /** Force rebuild all projects even if up-to-date */
  force?: boolean;
  /** Enable verbose logging */
  verbose?: boolean;
  /** Stop building on first error */
  stopBuildOnErrors?: boolean;
  /** Enable incremental compilation */
  incremental?: boolean;
  /** Assume changes only affect direct dependencies */
  assumeChangesOnlyAffectDirectDependencies?: boolean;
  /** Emit declaration files */
  declaration?: boolean;
  /** Emit declaration source maps */
  declarationMap?: boolean;
  /** Only emit declaration files */
  emitDeclarationOnly?: boolean;
  /** Emit source maps */
  sourceMap?: boolean;
  /** Emit inline source maps */
  inlineSourceMap?: boolean;
  /** Enable trace resolution logging */
  traceResolution?: boolean;
  /** Additional compiler options */
  [option: string]: CompilerOptionsValue | undefined;
}
```

### Solution Builder Host

Host interface providing file system operations for solution builder.

```typescript { .api }
interface SolutionBuilderHostBase<T extends BuilderProgram> extends ProgramHost<T> {
  /** Create a directory */
  createDirectory?(path: string): void;

  /** Write a file to disk */
  writeFile?(path: string, data: string, writeByteOrderMark?: boolean): void;

  /** Get custom transformers for a project */
  getCustomTransformers?: (project: string) => CustomTransformers | undefined;

  /** Get the last modified time of a file */
  getModifiedTime(fileName: string): Date | undefined;

  /** Set the modified time of a file */
  setModifiedTime(fileName: string, date: Date): void;

  /** Delete a file */
  deleteFile(fileName: string): void;

  /** Get parsed command line for a project */
  getParsedCommandLine?(fileName: string): ParsedCommandLine | undefined;

  /** Report diagnostic messages */
  reportDiagnostic: DiagnosticReporter;

  /** Report solution builder status messages */
  reportSolutionBuilderStatus: DiagnosticReporter;

  /** Hook called after program emit and diagnostics */
  afterProgramEmitAndDiagnostics?(program: T): void;
}

interface SolutionBuilderHost<T extends BuilderProgram> extends SolutionBuilderHostBase<T> {
  /** Report error summary */
  reportErrorSummary?: ReportEmitErrorSummary;
}

interface SolutionBuilderWithWatchHost<T extends BuilderProgram>
  extends SolutionBuilderHostBase<T>, WatchHost {}

type ReportEmitErrorSummary = (
  errorCount: number,
  filesInError: (ReportFileInError | undefined)[]
) => void;

interface ReportFileInError {
  fileName: string;
  line: number;
}
```

### Invalidated Projects

Interfaces for working with projects that need to be rebuilt.

```typescript { .api }
enum InvalidatedProjectKind {
  /** Project needs to be built */
  Build = 0,
  /** Only output file timestamps need updating */
  UpdateOutputFileStamps = 1
}

interface InvalidatedProjectBase {
  /** Type of invalidation */
  readonly kind: InvalidatedProjectKind;

  /** Resolved config file name */
  readonly project: ResolvedConfigFileName;

  /**
   * Complete building this project and update state
   * @param cancellationToken - Optional cancellation token
   * @param writeFile - Optional custom file writer
   * @param customTransformers - Optional custom transformers
   * @returns Exit status code
   */
  done(
    cancellationToken?: CancellationToken,
    writeFile?: WriteFileCallback,
    customTransformers?: CustomTransformers
  ): ExitStatus;

  /** Get compiler options for this project */
  getCompilerOptions(): CompilerOptions;

  /** Get current directory for this project */
  getCurrentDirectory(): string;
}

interface UpdateOutputFileStampsProject extends InvalidatedProjectBase {
  readonly kind: InvalidatedProjectKind.UpdateOutputFileStamps;

  /**
   * Update output file timestamps without rebuilding
   * @note Method name contains typo in TypeScript source (missing 'p' in "Stamps")
   */
  updateOutputFileStatmps(): void;
}

interface BuildInvalidedProject<T extends BuilderProgram> extends InvalidatedProjectBase {
  readonly kind: InvalidatedProjectKind.Build;

  /** Get the builder program */
  getBuilderProgram(): T | undefined;

  /** Get the program */
  getProgram(): Program | undefined;

  /** Get a source file by name */
  getSourceFile(fileName: string): SourceFile | undefined;

  /** Get all source files */
  getSourceFiles(): readonly SourceFile[];

  /** Get options diagnostics */
  getOptionsDiagnostics(cancellationToken?: CancellationToken): readonly Diagnostic[];

  /** Get global diagnostics */
  getGlobalDiagnostics(cancellationToken?: CancellationToken): readonly Diagnostic[];

  /** Get config file parsing diagnostics */
  getConfigFileParsingDiagnostics(): readonly Diagnostic[];

  /** Get syntactic diagnostics */
  getSyntacticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];

  /** Get all dependencies of a source file */
  getAllDependencies(sourceFile: SourceFile): readonly string[];

  /** Get semantic diagnostics */
  getSemanticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];

  /** Get semantic diagnostics of next affected file */
  getSemanticDiagnosticsOfNextAffectedFile(
    cancellationToken?: CancellationToken,
    ignoreSourceFile?: (sourceFile: SourceFile) => boolean
  ): AffectedFileResult<readonly Diagnostic[]>;

  /** Emit JavaScript and declaration files */
  emit(
    targetSourceFile?: SourceFile,
    writeFile?: WriteFileCallback,
    cancellationToken?: CancellationToken,
    emitOnlyDtsFiles?: boolean,
    customTransformers?: CustomTransformers
  ): EmitResult | undefined;
}

type InvalidatedProject<T extends BuilderProgram> =
  | UpdateOutputFileStampsProject
  | BuildInvalidedProject<T>;
```

### Utilities

Utility functions for working with solution builder.

```typescript { .api }
/**
 * Check if command line arguments indicate a build command
 * @param commandLineArgs - Command line arguments
 * @returns True if this is a --build command
 */
function isBuildCommand(commandLineArgs: readonly string[]): boolean;
```

**Usage Example:**

```typescript
import * as ts from 'typescript';

// Check if this is a build command
if (ts.isBuildCommand(process.argv.slice(2))) {
  // Parse as build command
  const parsedBuildCommand = ts.parseBuildCommand(process.argv.slice(2));
  // ... create solution builder and build
}
```
