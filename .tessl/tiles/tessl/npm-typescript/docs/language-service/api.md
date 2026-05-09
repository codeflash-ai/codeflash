# Language Service

Language Service APIs for editor integration features including code completions, diagnostics, refactorings, and navigation. The LanguageService interface provides 100+ methods for IDE functionality.

## Capabilities

### Language Service Creation

```typescript { .api }
function createLanguageService(
  host: LanguageServiceHost,
  documentRegistry?: DocumentRegistry,
  syntaxOnlyOrLanguageServiceMode?: boolean | LanguageServiceMode
): LanguageService;

function createDocumentRegistry(
  useCaseSensitiveFileNames?: boolean,
  currentDirectory?: string
): DocumentRegistry;

enum LanguageServiceMode {
  Semantic = 0,
  PartialSemantic = 1,
  Syntactic = 2
}

interface DocumentRegistry {
  acquireDocument(
    fileName: string,
    compilationSettingsOrHost: CompilerOptions | MinimalResolutionCacheHost,
    scriptSnapshot: IScriptSnapshot,
    version: string,
    scriptKind?: ScriptKind,
    sourceFileOptions?: CreateSourceFileOptions | ScriptTarget
  ): SourceFile;

  acquireDocumentWithKey(
    fileName: string,
    path: Path,
    compilationSettingsOrHost: CompilerOptions | MinimalResolutionCacheHost,
    key: DocumentRegistryBucketKey,
    scriptSnapshot: IScriptSnapshot,
    version: string,
    scriptKind?: ScriptKind,
    sourceFileOptions?: CreateSourceFileOptions | ScriptTarget
  ): SourceFile;

  updateDocument(
    fileName: string,
    compilationSettingsOrHost: CompilerOptions | MinimalResolutionCacheHost,
    scriptSnapshot: IScriptSnapshot,
    version: string,
    scriptKind?: ScriptKind,
    sourceFileOptions?: CreateSourceFileOptions | ScriptTarget
  ): SourceFile;

  updateDocumentWithKey(
    fileName: string,
    path: Path,
    compilationSettingsOrHost: CompilerOptions | MinimalResolutionCacheHost,
    key: DocumentRegistryBucketKey,
    scriptSnapshot: IScriptSnapshot,
    version: string,
    scriptKind?: ScriptKind,
    sourceFileOptions?: CreateSourceFileOptions | ScriptTarget
  ): SourceFile;

  releaseDocument(
    fileName: string,
    compilationSettings: CompilerOptions,
    scriptKind?: ScriptKind,
    impliedNodeFormat?: ResolutionMode
  ): void;

  releaseDocumentWithKey(
    path: Path,
    key: DocumentRegistryBucketKey,
    scriptKind?: ScriptKind,
    impliedNodeFormat?: ResolutionMode
  ): void;

  reportStats(): string;
}
```

### Language Service Interface

Main interface for editor functionality.

```typescript { .api }
interface LanguageService {
  /* Cleanup */
  cleanupSemanticCache(): void;
  dispose(): void;
  
  /* Diagnostics */
  getSyntacticDiagnostics(fileName: string): DiagnosticWithLocation[];
  getSemanticDiagnostics(fileName: string): Diagnostic[];
  getSuggestionDiagnostics(fileName: string): DiagnosticWithLocation[];
  getCompilerOptionsDiagnostics(): Diagnostic[];
  
  /* Completions */
  getCompletionsAtPosition(
    fileName: string,
    position: number,
    options: GetCompletionsAtPositionOptions | undefined,
    formattingSettings?: FormatCodeSettings
  ): WithMetadata<CompletionInfo> | undefined;
  
  getCompletionEntryDetails(
    fileName: string,
    position: number,
    entryName: string,
    formatOptions: FormatCodeOptions | FormatCodeSettings | undefined,
    source: string | undefined,
    preferences: UserPreferences | undefined,
    data: CompletionEntryData | undefined
  ): CompletionEntryDetails | undefined;
  
  getCompletionEntrySymbol(
    fileName: string,
    position: number,
    name: string,
    source: string | undefined
  ): Symbol | undefined;
  
  /* Quick Info */
  getQuickInfoAtPosition(
    fileName: string,
    position: number,
    maximumLength?: number
  ): QuickInfo | undefined;
  
  getNameOrDottedNameSpan(
    fileName: string,
    startPos: number,
    endPos: number
  ): TextSpan | undefined;
  
  getBreakpointStatementAtPosition(
    fileName: string,
    position: number
  ): TextSpan | undefined;
  
  /* Signature Help */
  getSignatureHelpItems(
    fileName: string,
    position: number,
    options: SignatureHelpItemsOptions | undefined
  ): SignatureHelpItems | undefined;
  
  /* Rename */
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
  
  getSmartSelectionRange(
    fileName: string,
    position: number
  ): SelectionRange;
  
  /* Definitions */
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
  
  /* References */
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
  
  /* Navigation */
  getNavigateToItems(
    searchValue: string,
    maxResultCount?: number,
    fileName?: string,
    excludeDtsFiles?: boolean,
    excludeLibFiles?: boolean
  ): NavigateToItem[];
  
  getNavigationBarItems(fileName: string): NavigationBarItem[];
  getNavigationTree(fileName: string): NavigationTree;
  
  /* Call Hierarchy */
  prepareCallHierarchy(
    fileName: string,
    position: number
  ): CallHierarchyItem | CallHierarchyItem[] | undefined;
  
  provideCallHierarchyIncomingCalls(
    fileName: string,
    position: number
  ): CallHierarchyIncomingCall[];
  
  provideCallHierarchyOutgoingCalls(
    fileName: string,
    position: number
  ): CallHierarchyOutgoingCall[];
  
  /* Inlay Hints */
  provideInlayHints(
    fileName: string,
    span: TextSpan,
    preferences: UserPreferences | undefined
  ): InlayHint[];
  
  /* Outlining */
  getOutliningSpans(fileName: string): OutliningSpan[];
  
  /* TODOs */
  getTodoComments(
    fileName: string,
    descriptors: TodoCommentDescriptor[]
  ): TodoComment[];
  
  /* Brace Matching */
  getBraceMatchingAtPosition(
    fileName: string,
    position: number
  ): TextSpan[];
  
  /* Indentation */
  getIndentationAtPosition(
    fileName: string,
    position: number,
    options: EditorOptions | EditorSettings
  ): number;
  
  /* Formatting */
  getFormattingEditsForRange(
    fileName: string,
    start: number,
    end: number,
    options: FormatCodeOptions | FormatCodeSettings
  ): TextChange[];
  
  getFormattingEditsForDocument(
    fileName: string,
    options: FormatCodeOptions | FormatCodeSettings
  ): TextChange[];
  
  getFormattingEditsAfterKeystroke(
    fileName: string,
    position: number,
    key: string,
    options: FormatCodeOptions | FormatCodeSettings
  ): TextChange[];
  
  /* JSDoc */
  getDocCommentTemplateAtPosition(
    fileName: string,
    position: number,
    options?: DocCommentTemplateOptions,
    formatOptions?: FormatCodeSettings
  ): TextInsertion | undefined;
  
  /* Brace Completion */
  isValidBraceCompletionAtPosition(
    fileName: string,
    position: number,
    openingBrace: number
  ): boolean;
  
  /* JSX */
  getJsxClosingTagAtPosition(
    fileName: string,
    position: number
  ): JsxClosingTagInfo | undefined;
  
  /* Linked Editing */
  getLinkedEditingRangeAtPosition(
    fileName: string,
    position: number
  ): LinkedEditingInfo | undefined;
  
  /* Span of Comment */
  getSpanOfEnclosingComment(
    fileName: string,
    position: number,
    onlyMultiLine: boolean
  ): TextSpan | undefined;
  
  /* Position Conversion */
  toLineColumnOffset?(
    fileName: string,
    position: number
  ): LineAndCharacter;
  
  /* Code Fixes */
  getCodeFixesAtPosition(
    fileName: string,
    start: number,
    end: number,
    errorCodes: readonly number[],
    formatOptions: FormatCodeSettings,
    preferences: UserPreferences
  ): readonly CodeFixAction[];
  
  getCombinedCodeFix(
    scope: CombinedCodeFixScope,
    fixId: {},
    formatOptions: FormatCodeSettings,
    preferences: UserPreferences
  ): CombinedCodeActions;
  
  applyCodeActionCommand(
    action: CodeActionCommand | CodeActionCommand[],
    formatSettings?: FormatCodeSettings
  ): Promise<ApplyCodeActionCommandResult | ApplyCodeActionCommandResult[]>;
  
  getSupportedCodeFixes(fileName?: string): readonly string[];
  
  /* Refactorings */
  getApplicableRefactors(
    fileName: string,
    positionOrRange: number | TextRange,
    preferences: UserPreferences | undefined,
    triggerReason?: RefactorTriggerReason,
    kind?: string,
    includeInteractiveActions?: boolean
  ): ApplicableRefactorInfo[];
  
  getEditsForRefactor(
    fileName: string,
    formatOptions: FormatCodeSettings,
    positionOrRange: number | TextRange,
    refactorName: string,
    actionName: string,
    preferences: UserPreferences | undefined,
    interactiveRefactorArguments?: InteractiveRefactorArguments
  ): RefactorEditInfo | undefined;
  
  getMoveToRefactoringFileSuggestions(
    fileName: string,
    positionOrRange: number | TextRange,
    preferences: UserPreferences | undefined,
    triggerReason?: RefactorTriggerReason,
    kind?: string
  ): { newFileName: string; files: string[] };
  
  /* Organize Imports */
  organizeImports(
    args: OrganizeImportsArgs,
    formatOptions: FormatCodeSettings,
    preferences: UserPreferences | undefined
  ): readonly FileTextChanges[];
  
  /* File Rename */
  getEditsForFileRename(
    oldFilePath: string,
    newFilePath: string,
    formatOptions: FormatCodeSettings,
    preferences: UserPreferences | undefined
  ): readonly FileTextChanges[];
  
  /* Emit */
  getEmitOutput(
    fileName: string,
    emitOnlyDtsFiles?: boolean,
    forceDtsEmit?: boolean
  ): EmitOutput;
  
  getProgram(): Program | undefined;
  
  /* Comments */
  toggleLineComment(fileName: string, textRange: TextRange): TextChange[];
  toggleMultilineComment(fileName: string, textRange: TextRange): TextChange[];
  commentSelection(fileName: string, textRange: TextRange): TextChange[];
  uncommentSelection(fileName: string, textRange: TextRange): TextChange[];
  
  /* Paste Edits */
  preparePasteEditsForFile(fileName: string, copiedTextRanges: TextRange[]): boolean;
  getPasteEdits(args: PasteEditsArgs, formatOptions: FormatCodeSettings): PasteEdits;
  
  /* Classifications */
  getSyntacticClassifications(
    fileName: string,
    span: TextSpan,
    format: SemanticClassificationFormat
  ): ClassifiedSpan[] | ClassifiedSpan2020[];
  
  getSemanticClassifications(
    fileName: string,
    span: TextSpan,
    format: SemanticClassificationFormat
  ): ClassifiedSpan[] | ClassifiedSpan2020[];
  
  getEncodedSyntacticClassifications(
    fileName: string,
    span: TextSpan
  ): Classifications;
  
  getEncodedSemanticClassifications(
    fileName: string,
    span: TextSpan,
    format?: SemanticClassificationFormat
  ): Classifications;
}
```

### Language Service Host

Host interface for file system operations.

```typescript { .api }
interface LanguageServiceHost extends GetEffectiveTypeRootsHost, MinimalResolutionCacheHost {
  getCompilationSettings(): CompilerOptions;
  getNewLine?(): string;
  getProjectVersion?(): string;
  getScriptFileNames(): string[];
  getScriptKind?(fileName: string): ScriptKind;
  getScriptVersion(fileName: string): string;
  getScriptSnapshot(fileName: string): IScriptSnapshot | undefined;
  getProjectReferences?(): readonly ProjectReference[] | undefined;
  getLocalizedDiagnosticMessages?(): any;
  getCancellationToken?(): HostCancellationToken;
  getCurrentDirectory(): string;
  getDefaultLibFileName(options: CompilerOptions): string;
  log?(s: string): void;
  trace?(s: string): void;
  error?(s: string): void;
  useCaseSensitiveFileNames?(): boolean;
  readDirectory?(
    path: string,
    extensions?: readonly string[],
    exclude?: readonly string[],
    include?: readonly string[],
    depth?: number
  ): string[];
  realpath?(path: string): string;
  readFile(path: string, encoding?: string): string | undefined;
  fileExists(path: string): boolean;
  getTypeRootsVersion?(): number;
  resolveModuleNameLiterals?(
    moduleLiterals: readonly StringLiteralLike[],
    containingFile: string,
    redirectedReference: ResolvedProjectReference | undefined,
    options: CompilerOptions,
    containingSourceFile: SourceFile,
    reusedNames: readonly StringLiteralLike[] | undefined
  ): readonly ResolvedModuleWithFailedLookupLocations[];
  getResolvedModuleWithFailedLookupLocationsFromCache?(
    modulename: string,
    containingFile: string,
    resolutionMode?: ResolutionMode
  ): ResolvedModuleWithFailedLookupLocations | undefined;
  resolveTypeReferenceDirectiveReferences?<T extends FileReference | string>(
    typeDirectiveReferences: readonly T[],
    containingFile: string,
    redirectedReference: ResolvedProjectReference | undefined,
    options: CompilerOptions,
    containingSourceFile: SourceFile | undefined,
    reusedNames: readonly T[] | undefined
  ): readonly ResolvedTypeReferenceDirectiveWithFailedLookupLocations[];
  getDirectories?(directoryName: string): string[];
  getCustomTransformers?(): CustomTransformers | undefined;
  isKnownTypesPackageName?(name: string): boolean;
  installPackage?(options: InstallPackageOptions): Promise<ApplyCodeActionCommandResult>;
  writeFile?(fileName: string, content: string): void;
  getParsedCommandLine?(fileName: string): ParsedCommandLine | undefined;
  jsDocParsingMode?: JSDocParsingMode | undefined;
}
```

### Completion Info

```typescript { .api }
interface CompletionInfo {
  flags?: CompletionInfoFlags;
  isGlobalCompletion: boolean;
  isMemberCompletion: boolean;
  isNewIdentifierLocation: boolean;
  optionalReplacementSpan?: TextSpan;
  isIncomplete?: boolean;
  entries: CompletionEntry[];
}

interface CompletionEntry {
  name: string;
  kind: ScriptElementKind;
  kindModifiers?: string;
  sortText: string;
  insertText?: string;
  filterText?: string;
  isSnippet?: boolean;
  replacementSpan?: TextSpan;
  hasAction?: true;
  source?: string;
  sourceDisplay?: SymbolDisplayPart[];
  labelDetails?: CompletionEntryLabelDetails;
  isRecommended?: true;
  isFromUncheckedFile?: true;
  isPackageJsonImport?: true;
  isImportStatementCompletion?: true;
  data?: CompletionEntryData;
}

interface CompletionEntryDetails {
  name: string;
  kind: ScriptElementKind;
  kindModifiers: string;
  displayParts: SymbolDisplayPart[];
  documentation?: SymbolDisplayPart[];
  tags?: JSDocTagInfo[];
  codeActions?: CodeAction[];
  source?: SymbolDisplayPart[];
  sourceDisplay?: SymbolDisplayPart[];
}

interface GetCompletionsAtPositionOptions extends UserPreferences {
  triggerCharacter?: CompletionsTriggerCharacter;
  triggerKind?: CompletionTriggerKind;
  includeSymbol?: boolean;
}

enum CompletionTriggerKind {
  Invoked = 1,
  TriggerCharacter = 2,
  TriggerForIncompleteCompletions = 3
}
```

### Signature Help

```typescript { .api }
interface SignatureHelpItems {
  items: SignatureHelpItem[];
  applicableSpan: TextSpan;
  selectedItemIndex: number;
  argumentIndex: number;
  argumentCount: number;
}

interface SignatureHelpItem {
  isVariadic: boolean;
  prefixDisplayParts: SymbolDisplayPart[];
  suffixDisplayParts: SymbolDisplayPart[];
  separatorDisplayParts: SymbolDisplayPart[];
  parameters: SignatureHelpParameter[];
  documentation: SymbolDisplayPart[];
  tags: JSDocTagInfo[];
}

interface SignatureHelpParameter {
  name: string;
  documentation: SymbolDisplayPart[];
  displayParts: SymbolDisplayPart[];
  isOptional: boolean;
  isRest?: boolean;
}
```

### Quick Info

```typescript { .api }
interface QuickInfo {
  kind: ScriptElementKind;
  kindModifiers: string;
  textSpan: TextSpan;
  displayParts?: SymbolDisplayPart[];
  documentation?: SymbolDisplayPart[];
  tags?: JSDocTagInfo[];
  canIncreaseVerbosityLevel?: boolean;
}
```

### Definition Info

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

### References

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

### Rename Info

```typescript { .api }
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

### Refactoring

```typescript { .api }
interface ApplicableRefactorInfo {
  name: string;
  description: string;
  actions: RefactorActionInfo[];
  inlineable?: boolean;
}

interface RefactorActionInfo {
  name: string;
  description: string;
  notApplicableReason?: string;
  kind?: string;
  isInteractive?: boolean;
}

interface RefactorEditInfo {
  edits: readonly FileTextChanges[];
  renameFilename?: string;
  renameLocation?: number;
  commands?: CodeActionCommand[];
}

type RefactorTriggerReason = "implicit" | "invoked";
```

### Code Actions

```typescript { .api }
interface CodeAction {
  description: string;
  changes: FileTextChanges[];
  commands?: CodeActionCommand[];
}

interface CodeFixAction extends CodeAction {
  fixName: string;
  fixId?: {};
  fixAllDescription?: string;
}

interface CombinedCodeActions {
  changes: readonly FileTextChanges[];
  commands?: readonly CodeActionCommand[];
}
```

### Navigation

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

### User Preferences

```typescript { .api }
interface UserPreferences {
  disableSuggestions?: boolean;
  quotePreference?: "auto" | "double" | "single";
  includeCompletionsForModuleExports?: boolean;
  includeCompletionsForImportStatements?: boolean;
  includeCompletionsWithSnippetText?: boolean;
  includeAutomaticOptionalChainCompletions?: boolean;
  includeCompletionsWithInsertText?: boolean;
  includeCompletionsWithClassMemberSnippets?: boolean;
  includeCompletionsWithObjectLiteralMethodSnippets?: boolean;
  useLabelDetailsInCompletionEntries?: boolean;
  allowIncompleteCompletions?: boolean;
  importModuleSpecifierPreference?: "shortest" | "project-relative" | "relative" | "non-relative";
  importModuleSpecifierEnding?: "auto" | "minimal" | "index" | "js";
  allowTextChangesInNewFiles?: boolean;
  providePrefixAndSuffixTextForRename?: boolean;
  provideRefactorNotApplicableReason?: boolean;
  includePackageJsonAutoImports?: "auto" | "on" | "off";
  jsxAttributeCompletionStyle?: "auto" | "braces" | "none";
  includeInlayParameterNameHints?: "none" | "literals" | "all";
  includeInlayParameterNameHintsWhenArgumentMatchesName?: boolean;
  includeInlayFunctionParameterTypeHints?: boolean;
  includeInlayVariableTypeHints?: boolean;
  includeInlayVariableTypeHintsWhenTypeMatchesName?: boolean;
  includeInlayPropertyDeclarationTypeHints?: boolean;
  includeInlayFunctionLikeReturnTypeHints?: boolean;
  includeInlayEnumMemberValueHints?: boolean;
  interactiveInlayHints?: boolean;
  allowRenameOfImportPath?: boolean;
  autoImportFileExcludePatterns?: string[];
  organizeImportsIgnoreCase?: "auto" | boolean;
  organizeImportsCollation?: "ordinal" | "unicode";
  organizeImportsLocale?: string;
  organizeImportsNumericCollation?: boolean;
  organizeImportsAccentCollation?: boolean;
  organizeImportsCaseFirst?: "upper" | "lower" | false;
  excludeLibrarySymbolsInNavTo?: boolean;
  lazyConfiguredProjectsFromExternalProject?: boolean;
  displayPartsForJSDoc?: boolean;
  generateReturnInDocTemplate?: boolean;
}
```
