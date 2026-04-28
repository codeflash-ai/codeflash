# Configuration Parsing

Parse TypeScript configuration files (tsconfig.json).

## Configuration Parsing APIs

```typescript { .api }
function findConfigFile(
  searchPath: string,
  fileExists: (fileName: string) => boolean,
  configName?: string
): string | undefined;

function parseJsonConfigFileContent(
  json: any,
  host: ParseConfigHost,
  basePath: string,
  existingOptions?: CompilerOptions,
  configFileName?: string,
  resolutionStack?: Path[],
  extraFileExtensions?: readonly FileExtensionInfo[],
  extendedConfigCache?: Map<string, ExtendedConfigCacheEntry>,
  existingWatchOptions?: WatchOptions
): ParsedCommandLine;

function parseJsonSourceFileConfigFileContent(
  sourceFile: TsConfigSourceFile,
  host: ParseConfigHost,
  basePath: string,
  existingOptions?: CompilerOptions,
  configFileName?: string,
  resolutionStack?: Path[],
  extraFileExtensions?: readonly FileExtensionInfo[],
  extendedConfigCache?: Map<string, ExtendedConfigCacheEntry>,
  existingWatchOptions?: WatchOptions
): ParsedCommandLine;

function readConfigFile(
  fileName: string,
  readFile: (path: string) => string | undefined
): { config?: any; error?: Diagnostic };

function getParsedCommandLineOfConfigFile(
  configFileName: string,
  optionsToExtend: CompilerOptions | undefined,
  host: ParseConfigFileHost,
  extendedConfigCache?: Map<string, ExtendedConfigCacheEntry>,
  watchOptionsToExtend?: WatchOptions,
  extraFileExtensions?: readonly FileExtensionInfo[]
): ParsedCommandLine | undefined;

interface ParsedCommandLine {
  options: CompilerOptions;
  typeAcquisition?: TypeAcquisition;
  fileNames: string[];
  projectReferences?: readonly ProjectReference[];
  watchOptions?: WatchOptions;
  raw?: any;
  errors: Diagnostic[];
  wildcardDirectories?: MapLike<WatchDirectoryFlags>;
  compileOnSave?: boolean;
}

interface ParseConfigHost {
  useCaseSensitiveFileNames: boolean;
  readDirectory(
    rootDir: string,
    extensions: readonly string[],
    excludes: readonly string[] | undefined,
    includes: readonly string[],
    depth?: number
  ): readonly string[];
  fileExists(path: string): boolean;
  readFile(path: string): string | undefined;
  trace?(s: string): void;
}
```

## Usage Example

```typescript
import * as ts from 'typescript';

// Find config file
const configPath = ts.findConfigFile(
  './src',
  ts.sys.fileExists
);

if (configPath) {
  // Read config file
  const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
  
  if (configFile.config) {
    // Parse config
    const parsed = ts.parseJsonConfigFileContent(
      configFile.config,
      {
        useCaseSensitiveFileNames: ts.sys.useCaseSensitiveFileNames,
        readDirectory: ts.sys.readDirectory,
        fileExists: ts.sys.fileExists,
        readFile: ts.sys.readFile
      },
      ts.getDirectoryPath(configPath)
    );
    
    // Use parsed options
    const program = ts.createProgram({
      rootNames: parsed.fileNames,
      options: parsed.options
    });
  }
}
```

