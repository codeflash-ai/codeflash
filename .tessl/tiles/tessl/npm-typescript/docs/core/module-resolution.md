# Module Resolution

APIs for resolving module imports and type reference directives with support for multiple resolution strategies.

## Capabilities

### Module Name Resolution

Resolve module names to file paths.

```typescript { .api }
function resolveModuleName(
  moduleName: string,
  containingFile: string,
  compilerOptions: CompilerOptions,
  host: ModuleResolutionHost,
  cache?: ModuleResolutionCache,
  redirectedReference?: ResolvedProjectReference,
  resolutionMode?: ResolutionMode
): ResolvedModuleWithFailedLookupLocations;

function nodeModuleNameResolver(
  moduleName: string,
  containingFile: string,
  compilerOptions: CompilerOptions,
  host: ModuleResolutionHost,
  cache?: ModuleResolutionCache,
  redirectedReference?: ResolvedProjectReference
): ResolvedModuleWithFailedLookupLocations;

function classicNameResolver(
  moduleName: string,
  containingFile: string,
  compilerOptions: CompilerOptions,
  host: ModuleResolutionHost,
  cache?: NonRelativeModuleNameResolutionCache | undefined,
  redirectedReference?: ResolvedProjectReference
): ResolvedModuleWithFailedLookupLocations;

function bundlerModuleNameResolver(
  moduleName: string,
  containingFile: string,
  compilerOptions: CompilerOptions,
  host: ModuleResolutionHost,
  cache?: ModuleResolutionCache,
  redirectedReference?: ResolvedProjectReference
): ResolvedModuleWithFailedLookupLocations;
```

### Type Reference Directive Resolution

Resolve type reference directives.

```typescript { .api }
function resolveTypeReferenceDirective(
  typeReferenceDirectiveName: string,
  containingFile: string | undefined,
  compilerOptions: CompilerOptions,
  host: ModuleResolutionHost,
  redirectedReference?: ResolvedProjectReference,
  cache?: TypeReferenceDirectiveResolutionCache,
  resolutionMode?: ResolutionMode
): ResolvedTypeReferenceDirectiveWithFailedLookupLocations;
```

### Module Resolution Host

Interface for file system operations during resolution.

```typescript { .api }
interface ModuleResolutionHost {
  fileExists(fileName: string): boolean;
  readFile(fileName: string): string | undefined;
  trace?(s: string): void;
  directoryExists?(directoryName: string): boolean;
  realpath?(path: string): string;
  getCurrentDirectory?(): string;
  getDirectories?(path: string): string[];
  useCaseSensitiveFileNames?: boolean | (() => boolean) | undefined;
}
```

### Resolved Module

Result of module resolution.

```typescript { .api }
interface ResolvedModule {
  resolvedFileName: string;
  isExternalLibraryImport?: boolean;
  resolvedUsingTsExtension?: boolean;
}

interface ResolvedModuleFull extends ResolvedModule {
  extension: string;
  packageId?: PackageId;
}

interface ResolvedModuleWithFailedLookupLocations {
  readonly resolvedModule: ResolvedModuleFull | undefined;
}

interface PackageId {
  name: string;
  subModuleName: string;
  version: string;
}
```

### Type Reference Directive

```typescript { .api }
interface ResolvedTypeReferenceDirective {
  primary: boolean;
  resolvedFileName: string | undefined;
  packageId?: PackageId;
  isExternalLibraryImport?: boolean;
}

interface ResolvedTypeReferenceDirectiveWithFailedLookupLocations {
  readonly resolvedTypeReferenceDirective: ResolvedTypeReferenceDirective | undefined;
}
```

### Module Resolution Cache

Cache for module resolution results.

```typescript { .api }
function createModuleResolutionCache(
  currentDirectory: string,
  getCanonicalFileName: (s: string) => string,
  options?: CompilerOptions
): ModuleResolutionCache;

interface ModuleResolutionCache {
  getPackageJsonInfoCache(): PackageJsonInfoCache;
}

function createTypeReferenceDirectiveResolutionCache(
  currentDirectory: string,
  getCanonicalFileName: (s: string) => string,
  options?: CompilerOptions,
  packageJsonInfoCache?: PackageJsonInfoCache
): TypeReferenceDirectiveResolutionCache;

interface TypeReferenceDirectiveResolutionCache {
  getOrCreateCacheForDirectory(directoryName: string, redirectedReference?: ResolvedProjectReference): PerDirectoryResolutionCache<ResolvedTypeReferenceDirectiveWithFailedLookupLocations>;
}
```

### File Extensions

```typescript { .api }
enum Extension {
  Ts = ".ts",
  Tsx = ".tsx",
  Dts = ".d.ts",
  Js = ".js",
  Jsx = ".jsx",
  Json = ".json",
  TsBuildInfo = ".tsbuildinfo",
  Mjs = ".mjs",
  Mts = ".mts",
  Dmts = ".d.mts",
  Cjs = ".cjs",
  Cts = ".cts",
  Dcts = ".d.cts"
}

interface FileExtensionInfo {
  extension: string;
  isMixedContent: boolean;
  scriptKind?: ScriptKind;
}
```

### Resolution Mode

```typescript { .api }
type ResolutionMode = ModuleKind.CommonJS | ModuleKind.ESNext;

function getModeForFileReference(ref: FileReference | string, containingFileMode: ResolutionMode): ResolutionMode;
function getModeForUsageLocation(file: SourceFile, usage: StringLiteralLike): ResolutionMode;
function getModeForResolutionAtIndex(file: SourceFile, index: number): ResolutionMode;
```

### Type Roots

```typescript { .api }
function getEffectiveTypeRoots(
  options: CompilerOptions,
  host: GetEffectiveTypeRootsHost
): string[] | undefined;

interface GetEffectiveTypeRootsHost {
  directoryExists?(directoryName: string): boolean;
  getCurrentDirectory?(): string;
}
```

### Automatic Type Acquisition

```typescript { .api }
function getAutomaticTypeDirectiveNames(
  options: CompilerOptions,
  host: ModuleResolutionHost
): string[];
```

## Usage Example

Resolving a module name:

```typescript
import * as ts from 'typescript';

const compilerOptions: ts.CompilerOptions = {
  moduleResolution: ts.ModuleResolutionKind.NodeJs,
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.CommonJS
};

const host: ts.ModuleResolutionHost = {
  fileExists: (fileName: string) => ts.sys.fileExists(fileName),
  readFile: (fileName: string) => ts.sys.readFile(fileName)
};

const result = ts.resolveModuleName(
  'lodash',
  '/project/src/index.ts',
  compilerOptions,
  host
);

if (result.resolvedModule) {
  console.log('Resolved to:', result.resolvedModule.resolvedFileName);
  console.log('Extension:', result.resolvedModule.extension);
  console.log('Is external:', result.resolvedModule.isExternalLibraryImport);
}
```
