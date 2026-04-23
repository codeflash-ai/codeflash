# Code Completions

APIs for providing code completion suggestions in editors.

## Completion APIs

```typescript { .api }
interface LanguageService {
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
}
```

## Completion Info

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

## Signature Help

```typescript { .api }
interface LanguageService {
  getSignatureHelpItems(
    fileName: string,
    position: number,
    options: SignatureHelpItemsOptions | undefined
  ): SignatureHelpItems | undefined;
}

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
  tags?: JSDocTagInfo[];
}

interface SignatureHelpParameter {
  name: string;
  documentation: SymbolDisplayPart[];
  displayParts: SymbolDisplayPart[];
  isOptional: boolean;
  isRest?: boolean;
}
```

## Quick Info

```typescript { .api }
interface LanguageService {
  getQuickInfoAtPosition(
    fileName: string,
    position: number,
    maximumLength?: number
  ): QuickInfo | undefined;
}

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

