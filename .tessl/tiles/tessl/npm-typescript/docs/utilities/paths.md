# Path Utilities

Helper functions for path manipulation and file operations.

## Path Functions

```typescript { .api }
function getDefaultLibFilePath(options: CompilerOptions): string;

function normalizePath(path: string): string;

function normalizeSlashes(path: string): string;

function getRootLength(path: string): number;

function getDirectoryPath(path: Path): Path;
function getDirectoryPath(path: string): string;

function isUrl(path: string): boolean;

function pathIsRelative(path: string): boolean;

function getBaseFileName(path: string): string;
function getBaseFileName(
  path: string,
  extensions: string | readonly string[],
  ignoreCase?: boolean
): string;

function combinePaths(path: string, ...paths: (string | undefined)[]): string;

function resolvePath(path: string, ...paths: (string | undefined)[]): string;

function getRelativePathToDirectoryOrUrl(
  directoryPathOrUrl: string,
  relativeOrAbsolutePath: string,
  currentDirectory: string,
  getCanonicalFileName: (fileName: string) => string,
  isAbsolutePathAnUrl: boolean
): string;
```

## Usage Example

```typescript
import * as ts from 'typescript';

const normalized = ts.normalizePath('src\\components\\Button.tsx');
console.log(normalized); // src/components/Button.tsx

const fullPath = ts.combinePaths('/project', 'src', 'index.ts');
console.log(fullPath); // /project/src/index.ts

const dirPath = ts.getDirectoryPath('/project/src/index.ts');
console.log(dirPath); // /project/src

const fileName = ts.getBaseFileName('/project/src/index.ts');
console.log(fileName); // index.ts

const baseName = ts.getBaseFileName('/project/src/index.ts', '.ts');
console.log(baseName); // index

console.log(ts.pathIsRelative('./file.ts'));    // true
console.log(ts.pathIsRelative('../file.ts'));   // true
console.log(ts.pathIsRelative('/abs/file.ts')); // false
```

