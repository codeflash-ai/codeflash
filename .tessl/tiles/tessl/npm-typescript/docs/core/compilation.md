# Core Compilation APIs

Core TypeScript compiler APIs for program creation, compilation, and code generation.

## Program Creation

Create and manage TypeScript programs.

```typescript { .api }
function createProgram(
  rootNames: readonly string[],
  options: CompilerOptions,
  host?: CompilerHost,
  oldProgram?: Program,
  configFileParsingDiagnostics?: readonly Diagnostic[]
): Program;

function createProgram(createProgramOptions: CreateProgramOptions): Program;

interface CreateProgramOptions {
  rootNames: readonly string[];
  options: CompilerOptions;
  projectReferences?: readonly ProjectReference[];
  host?: CompilerHost;
  oldProgram?: Program;
  configFileParsingDiagnostics?: readonly Diagnostic[];
}
```

**Usage Example:**

```typescript
import * as ts from 'typescript';

const options: ts.CompilerOptions = {
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.CommonJS,
  strict: true,
  esModuleInterop: true
};

const program = ts.createProgram({
  rootNames: ['index.ts', 'utils.ts'],
  options
});
```

## Program Interface

Main program interface for compilation operations.

```typescript { .api }
interface Program {
  getRootFileNames(): readonly string[];
  getSourceFiles(): readonly SourceFile[];
  getSourceFile(fileName: string): SourceFile | undefined;
  getTypeChecker(): TypeChecker;
  getNodeCount(): number;
  getIdentifierCount(): number;
  getSymbolCount(): number;
  getTypeCount(): number;
  getInstantiationCount(): number;
  getRelationCacheSizes(): {
    assignable: number;
    identity: number;
    subtype: number;
    strictSubtype: number;
  };
  emit(
    targetSourceFile?: SourceFile,
    writeFile?: WriteFileCallback,
    cancellationToken?: CancellationToken,
    emitOnlyDtsFiles?: boolean,
    customTransformers?: CustomTransformers
  ): EmitResult;
  getSemanticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
  getSyntacticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly DiagnosticWithLocation[];
  getDeclarationDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly DiagnosticWithLocation[];
  getGlobalDiagnostics(cancellationToken?: CancellationToken): readonly Diagnostic[];
  getConfigFileParsingDiagnostics(): readonly Diagnostic[];
  getCompilerOptions(): CompilerOptions;
  getCurrentDirectory(): string;
  getOptionsDiagnostics(cancellationToken?: CancellationToken): readonly Diagnostic[];
  getProjectReferences(): readonly ProjectReference[] | undefined;
  getResolvedProjectReferences(): readonly (ResolvedProjectReference | undefined)[] | undefined;
  isSourceFileFromExternalLibrary(file: SourceFile): boolean;
  isSourceFileDefaultLibrary(file: SourceFile): boolean;
  getModeForUsageLocation(file: SourceFile, usage: StringLiteralLike): ResolutionMode;
  getModeForResolutionAtIndex(file: SourceFile, index: number): ResolutionMode;
}
```

## Compiler Options

Comprehensive compiler configuration options.

```typescript { .api }
interface CompilerOptions {
  /* Language and Environment */
  target?: ScriptTarget;
  lib?: string[];
  jsx?: JsxEmit;
  experimentalDecorators?: boolean;
  emitDecoratorMetadata?: boolean;
  jsxFactory?: string;
  jsxFragmentFactory?: string;
  jsxImportSource?: string;
  reactNamespace?: string;
  noLib?: boolean;
  useDefineForClassFields?: boolean;
  moduleDetection?: ModuleDetectionKind;
  
  /* Modules */
  module?: ModuleKind;
  moduleResolution?: ModuleResolutionKind;
  moduleSuffixes?: string[];
  noResolve?: boolean;
  paths?: MapLike<string[]>;
  rootDirs?: string[];
  rootDir?: string;
  typeRoots?: string[];
  types?: string[];
  allowArbitraryExtensions?: boolean;
  allowImportingTsExtensions?: boolean;
  allowUmdGlobalAccess?: boolean;
  baseUrl?: string;
  customConditions?: string[];
  resolveJsonModule?: boolean;
  resolvePackageJsonExports?: boolean;
  resolvePackageJsonImports?: boolean;
  rewriteRelativeImportExtensions?: boolean;
  noUncheckedSideEffectImports?: boolean;

  /* JavaScript Support */
  allowJs?: boolean;
  checkJs?: boolean;
  maxNodeModuleJsDepth?: number;
  
  /* Emit */
  declaration?: boolean;
  declarationMap?: boolean;
  declarationDir?: string;
  emitDeclarationOnly?: boolean;
  downlevelIteration?: boolean;
  emitBOM?: boolean;
  importHelpers?: boolean;
  inlineSourceMap?: boolean;
  inlineSources?: boolean;
  mapRoot?: string;
  newLine?: NewLineKind;
  noEmit?: boolean;
  noEmitHelpers?: boolean;
  noEmitOnError?: boolean;
  outDir?: string;
  outFile?: string;
  preserveConstEnums?: boolean;
  preserveValueImports?: boolean;
  removeComments?: boolean;
  sourceMap?: boolean;
  sourceRoot?: string;
  stripInternal?: boolean;
  
  /* Interop Constraints */
  allowSyntheticDefaultImports?: boolean;
  esModuleInterop?: boolean;
  forceConsistentCasingInFileNames?: boolean;
  isolatedModules?: boolean;
  isolatedDeclarations?: boolean;
  preserveSymlinks?: boolean;
  verbatimModuleSyntax?: boolean;
  
  /* Type Checking */
  strict?: boolean;
  alwaysStrict?: boolean;
  exactOptionalPropertyTypes?: boolean;
  noFallthroughCasesInSwitch?: boolean;
  noImplicitAny?: boolean;
  noImplicitOverride?: boolean;
  noImplicitReturns?: boolean;
  noImplicitThis?: boolean;
  noPropertyAccessFromIndexSignature?: boolean;
  noUncheckedIndexedAccess?: boolean;
  noUnusedLocals?: boolean;
  noUnusedParameters?: boolean;
  strictBindCallApply?: boolean;
  strictBuiltinIteratorReturn?: boolean;
  strictFunctionTypes?: boolean;
  strictNullChecks?: boolean;
  strictPropertyInitialization?: boolean;
  useUnknownInCatchVariables?: boolean;
  
  /* Completeness */
  skipDefaultLibCheck?: boolean;
  skipLibCheck?: boolean;
  
  /* Project References */
  composite?: boolean;
  disableReferencedProjectLoad?: boolean;
  disableSolutionSearching?: boolean;
  disableSourceOfProjectReferenceRedirect?: boolean;
  incremental?: boolean;
  tsBuildInfoFile?: string;
  
  /* Advanced */
  allowUnreachableCode?: boolean;
  allowUnusedLabels?: boolean;
  assumeChangesOnlyAffectDirectDependencies?: boolean;
  disableSizeLimit?: boolean;
  noCheck?: boolean;
  noErrorTruncation?: boolean;
  preserveWatchOutput?: boolean;
  suppressExcessPropertyErrors?: boolean;
  suppressImplicitAnyIndexErrors?: boolean;
  traceResolution?: boolean;
  ignoreDeprecations?: string;

  [option: string]: CompilerOptionsValue | TsConfigSourceFile | undefined;
}

type CompilerOptionsValue = 
  | string 
  | number 
  | boolean 
  | (string | number)[] 
  | string[] 
  | MapLike<string[]> 
  | PluginImport[] 
  | ProjectReference[] 
  | null 
  | undefined;
```

## Compiler Host

Interface for file system operations during compilation.

```typescript { .api }
interface CompilerHost extends ModuleResolutionHost {
  getSourceFile(
    fileName: string,
    languageVersionOrOptions: ScriptTarget | CreateSourceFileOptions,
    onError?: (message: string) => void,
    shouldCreateNewSourceFile?: boolean
  ): SourceFile | undefined;
  
  getSourceFileByPath?(
    fileName: string,
    path: Path,
    languageVersionOrOptions: ScriptTarget | CreateSourceFileOptions,
    onError?: (message: string) => void,
    shouldCreateNewSourceFile?: boolean
  ): SourceFile | undefined;
  
  getCancellationToken?(): CancellationToken;
  getDefaultLibFileName(options: CompilerOptions): string;
  getDefaultLibLocation?(): string;
  writeFile: WriteFileCallback;
  getCurrentDirectory(): string;
  getCanonicalFileName(fileName: string): string;
  useCaseSensitiveFileNames(): boolean;
  getNewLine(): string;
  readDirectory?(
    rootDir: string,
    extensions: readonly string[],
    excludes: readonly string[] | undefined,
    includes: readonly string[],
    depth?: number
  ): string[];
  resolveModuleNameLiterals?(
    moduleLiterals: readonly StringLiteralLike[],
    containingFile: string,
    redirectedReference: ResolvedProjectReference | undefined,
    options: CompilerOptions,
    containingSourceFile: SourceFile,
    reusedNames: readonly StringLiteralLike[] | undefined
  ): readonly ResolvedModuleWithFailedLookupLocations[];
  resolveTypeReferenceDirectiveReferences?<T extends FileReference | string>(
    typeDirectiveReferences: readonly T[],
    containingFile: string,
    redirectedReference: ResolvedProjectReference | undefined,
    options: CompilerOptions,
    containingSourceFile: SourceFile | undefined,
    reusedNames: readonly T[] | undefined
  ): readonly ResolvedTypeReferenceDirectiveWithFailedLookupLocations[];
  getEnvironmentVariable?(name: string): string | undefined;
  hasInvalidatedResolutions?(filePath: Path): boolean;
  createHash?(data: string): string;
  getParsedCommandLine?(fileName: string): ParsedCommandLine | undefined;
  jsDocParsingMode?: JSDocParsingMode;
}

type WriteFileCallback = (
  fileName: string,
  text: string,
  writeByteOrderMark: boolean,
  onError?: (message: string) => void,
  sourceFiles?: readonly SourceFile[],
  data?: WriteFileCallbackData
) => void;

function createCompilerHost(
  options: CompilerOptions,
  setParentNodes?: boolean
): CompilerHost;
```

## Emit Result

Result of program emit operation.

```typescript { .api }
interface EmitResult {
  emitSkipped: boolean;
  diagnostics: readonly Diagnostic[];
  emittedFiles?: string[];
}

interface EmitOutput {
  outputFiles: OutputFile[];
  emitSkipped: boolean;
}

interface OutputFile {
  name: string;
  writeByteOrderMark: boolean;
  text: string;
}
```

## Incremental Compilation

Incremental compilation APIs enable faster rebuilds by reusing previous compilation state.

```typescript { .api }
function createIncrementalCompilerHost(
  options: CompilerOptions,
  system?: System
): CompilerHost;

function createIncrementalProgram<T extends BuilderProgram = EmitAndSemanticDiagnosticsBuilderProgram>(
  options: IncrementalProgramOptions<T>
): T;

interface IncrementalProgramOptions<T extends BuilderProgram> {
  rootNames: readonly string[];
  options: CompilerOptions;
  configFileParsingDiagnostics?: readonly Diagnostic[];
  projectReferences?: readonly ProjectReference[];
  host?: CompilerHost;
  createProgram?: CreateProgram<T>;
}

function readBuilderProgram(
  compilerOptions: CompilerOptions,
  host: ReadBuildProgramHost
): EmitAndSemanticDiagnosticsBuilderProgram | undefined;

interface ReadBuildProgramHost {
  useCaseSensitiveFileNames(): boolean;
  getCurrentDirectory(): string;
  readFile(fileName: string): string | undefined;
}

function createEmitAndSemanticDiagnosticsBuilderProgram(
  rootNames: readonly string[] | undefined,
  options: CompilerOptions | undefined,
  host?: CompilerHost,
  oldProgram?: EmitAndSemanticDiagnosticsBuilderProgram,
  configFileParsingDiagnostics?: readonly Diagnostic[],
  projectReferences?: readonly ProjectReference[]
): EmitAndSemanticDiagnosticsBuilderProgram;

function createSemanticDiagnosticsBuilderProgram(
  rootNames: readonly string[] | undefined,
  options: CompilerOptions | undefined,
  host?: CompilerHost,
  oldProgram?: SemanticDiagnosticsBuilderProgram,
  configFileParsingDiagnostics?: readonly Diagnostic[],
  projectReferences?: readonly ProjectReference[]
): SemanticDiagnosticsBuilderProgram;

interface BuilderProgram {
  getProgram(): Program;
  getCompilerOptions(): CompilerOptions;
  getSourceFile(fileName: string): SourceFile | undefined;
  getSourceFiles(): readonly SourceFile[];
  getCurrentDirectory(): string;
}

interface EmitAndSemanticDiagnosticsBuilderProgram extends BuilderProgram {
  getSemanticDiagnosticsOfNextAffectedFile(
    cancellationToken?: CancellationToken,
    ignoreSourceFile?: (sourceFile: SourceFile) => boolean
  ): AffectedFileResult<readonly Diagnostic[]>;
  emit(
    targetSourceFile?: SourceFile,
    writeFile?: WriteFileCallback,
    cancellationToken?: CancellationToken,
    emitOnlyDtsFiles?: boolean,
    customTransformers?: CustomTransformers
  ): EmitResult | undefined;
  getAllDependencies(sourceFile: SourceFile): readonly string[];
  getSyntacticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
  getSemanticDiagnostics(
    sourceFile?: SourceFile,
    cancellationToken?: CancellationToken
  ): readonly Diagnostic[];
}
```

**Usage Example:**

```typescript
import * as ts from 'typescript';

const host = ts.createIncrementalCompilerHost({
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.CommonJS,
  incremental: true,
  tsBuildInfoFile: '.tsbuildinfo'
});

const program = ts.createIncrementalProgram({
  rootNames: ['index.ts', 'utils.ts'],
  options: {
    target: ts.ScriptTarget.ES2020,
    module: ts.ModuleKind.CommonJS,
    incremental: true
  },
  host
});

let result = program.emit();
```

## Watch Program

Watch mode compilation.

```typescript { .api }
function createWatchCompilerHost<T extends BuilderProgram>(
  rootFiles: string[],
  options: CompilerOptions,
  system: System,
  createProgram?: CreateProgram<T>,
  reportDiagnostic?: DiagnosticReporter,
  reportWatchStatus?: WatchStatusReporter,
  watchOptionsToExtend?: WatchOptions,
  extraFileExtensions?: readonly FileExtensionInfo[]
): WatchCompilerHostOfFilesAndCompilerOptions<T>;

function createWatchProgram<T extends BuilderProgram>(
  host: WatchCompilerHostOfFilesAndCompilerOptions<T>
): WatchOfFilesAndCompilerOptions<T>;

interface WatchCompilerHost<T extends BuilderProgram> {
  afterProgramCreate?(program: T): void;
  useCaseSensitiveFileNames(): boolean;
  getNewLine(): string;
  getCurrentDirectory(): string;
  getDefaultLibFileName(options: CompilerOptions): string;
  getDefaultLibLocation?(): string;
  createHash?(data: string): string;
  fileExists(path: string): boolean;
  readFile(path: string, encoding?: string): string | undefined;
  directoryExists?(path: string): boolean;
  getDirectories?(path: string): string[];
  readDirectory?(
    path: string,
    extensions?: readonly string[],
    exclude?: readonly string[],
    include?: readonly string[],
    depth?: number
  ): string[];
  watchFile(
    path: string,
    callback: FileWatcherCallback,
    pollingInterval?: number,
    options?: WatchOptions
  ): FileWatcher;
  watchDirectory(
    path: string,
    callback: DirectoryWatcherCallback,
    recursive?: boolean,
    options?: WatchOptions
  ): FileWatcher;
  setTimeout?(callback: (...args: any[]) => void, ms: number, ...args: any[]): any;
  clearTimeout?(timeoutId: any): void;
}
```

## Project References

Multi-project compilation support.

```typescript { .api }
interface ProjectReference {
  path: string;
  originalPath?: string;
  prepend?: boolean;
  circular?: boolean;
}

interface ResolvedProjectReference {
  commandLine: ParsedCommandLine;
  sourceFile: SourceFile;
  references?: readonly (ResolvedProjectReference | undefined)[];
}
```

## Enumerations

```typescript { .api }
enum ScriptTarget {
  ES3 = 0,
  ES5 = 1,
  ES2015 = 2,
  ES2016 = 3,
  ES2017 = 4,
  ES2018 = 5,
  ES2019 = 6,
  ES2020 = 7,
  ES2021 = 8,
  ES2022 = 9,
  ES2023 = 10,
  ES2024 = 11,
  ESNext = 99,
  JSON = 100,
  Latest = 99
}

enum ModuleKind {
  None = 0,
  CommonJS = 1,
  AMD = 2,
  UMD = 3,
  System = 4,
  ES2015 = 5,
  ES2020 = 6,
  ES2022 = 7,
  ESNext = 99,
  Node16 = 100,
  Node18 = 101,
  Node20 = 102,
  NodeNext = 199,
  Preserve = 200
}

enum ModuleResolutionKind {
  Classic = 1,
  NodeJs = 2,
  Node16 = 3,
  NodeNext = 99,
  Bundler = 100
}

enum ModuleDetectionKind {
  Legacy = 1,
  Auto = 2,
  Force = 3
}

enum JsxEmit {
  None = 0,
  Preserve = 1,
  React = 2,
  ReactNative = 3,
  ReactJSX = 4,
  ReactJSXDev = 5
}

enum NewLineKind {
  CarriageReturnLineFeed = 0,
  LineFeed = 1
}
```

