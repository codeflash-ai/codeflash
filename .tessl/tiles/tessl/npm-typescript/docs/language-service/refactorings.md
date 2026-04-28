# Code Refactorings

APIs for code refactoring operations like extract function, move to file, etc.

## Refactoring APIs

```typescript { .api }
interface LanguageService {
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
}
```

## Refactoring Interfaces

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

## Code Actions

```typescript { .api }
interface LanguageService {
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
}
```

## Code Action Interfaces

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

