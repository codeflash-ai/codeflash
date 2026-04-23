# Navigation APIs

APIs for navigating code: go to definition, find references, and symbol navigation.

## Definition and References

```typescript { .api }
interface LanguageService {
  getDefinitionAtPosition(
    fileName: string,
    position: number
  ): readonly DefinitionInfo[] | undefined;
  
  getDefinitionAndBoundSpan(
    fileName: string,
    position: number
  ): DefinitionInfoAndBoundSpan | undefined;
  
  getTypeDefinitionAtPosition(
    fileName: string,
    position: number
  ): readonly DefinitionInfo[] | undefined;
  
  getImplementationAtPosition(
    fileName: string,
    position: number
  ): readonly ImplementationLocation[] | undefined;
  
  getReferencesAtPosition(
    fileName: string,
    position: number
  ): ReferenceEntry[] | undefined;
  
  findReferences(
    fileName: string,
    position: number
  ): ReferencedSymbol[] | undefined;
  
  getDocumentHighlights(
    fileName: string,
    position: number,
    filesToSearch: string[]
  ): DocumentHighlights[] | undefined;
  
  getFileReferences(fileName: string): ReferenceEntry[];
}
```

## Definition Info

```typescript { .api }
interface DefinitionInfo extends DocumentSpan {
  kind: ScriptElementKind;
  name: string;
  containerKind: ScriptElementKind;
  containerName: string;
  unverified?: boolean;
}

interface DefinitionInfoAndBoundSpan {
  definitions?: readonly DefinitionInfo[];
  textSpan: TextSpan;
}

interface DocumentSpan {
  textSpan: TextSpan;
  fileName: string;
  originalTextSpan?: TextSpan;
  originalFileName?: string;
  contextSpan?: TextSpan;
  originalContextSpan?: TextSpan;
}
```

## References

```typescript { .api }
interface ReferenceEntry extends DocumentSpan {
  isWriteAccess: boolean;
  isDefinition: boolean;
  isInString?: true;
}

interface ReferencedSymbol {
  definition: ReferencedSymbolDefinitionInfo;
  references: ReferencedSymbolEntry[];
}

interface ReferencedSymbolDefinitionInfo extends DefinitionInfo {
  displayParts: SymbolDisplayPart[];
}

interface ReferencedSymbolEntry extends ReferenceEntry {
  isInString?: true;
}
```

## Navigation Bar and Tree

```typescript { .api }
interface LanguageService {
  getNavigateToItems(
    searchValue: string,
    maxResultCount?: number,
    fileName?: string,
    excludeDtsFiles?: boolean,
    excludeLibFiles?: boolean
  ): NavigateToItem[];
  
  getNavigationBarItems(fileName: string): NavigationBarItem[];
  getNavigationTree(fileName: string): NavigationTree;
}
```

## Navigation Interfaces

```typescript { .api }
interface NavigationBarItem {
  text: string;
  kind: ScriptElementKind;
  kindModifiers: string;
  spans: TextSpan[];
  childItems: NavigationBarItem[];
  indent: number;
  bolded: boolean;
  grayed: boolean;
}

interface NavigationTree {
  text: string;
  kind: ScriptElementKind;
  kindModifiers: string;
  spans: TextSpan[];
  nameSpan: TextSpan | undefined;
  childItems?: NavigationTree[];
}

interface NavigateToItem {
  name: string;
  kind: ScriptElementKind;
  kindModifiers: string;
  matchKind: "exact" | "prefix" | "substring" | "camelCase";
  isCaseSensitive: boolean;
  fileName: string;
  textSpan: TextSpan;
  containerName: string;
  containerKind: ScriptElementKind;
}
```

## Rename

```typescript { .api }
interface LanguageService {
  getRenameInfo(
    fileName: string,
    position: number,
    preferences: UserPreferences
  ): RenameInfo;
  
  findRenameLocations(
    fileName: string,
    position: number,
    findInStrings: boolean,
    findInComments: boolean,
    preferences: UserPreferences
  ): readonly RenameLocation[] | undefined;
}

type RenameInfo = RenameInfoSuccess | RenameInfoFailure;

interface RenameInfoSuccess {
  canRename: true;
  fileToRename?: string;
  displayName: string;
  fullDisplayName: string;
  kind: ScriptElementKind;
  kindModifiers: string;
  triggerSpan: TextSpan;
}

interface RenameInfoFailure {
  canRename: false;
  localizedErrorMessage: string;
}

interface RenameLocation extends DocumentSpan {
  prefixText?: string;
  suffixText?: string;
}
```

