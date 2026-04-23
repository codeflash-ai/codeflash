# TSServer Protocol

TypeScript server protocol for editor-compiler communication. Defines request/response messages and events for the TypeScript language server.

## Capabilities

### Protocol Namespace

```typescript { .api }
namespace ts.server.protocol {
  // Protocol types and interfaces
}
```

### Message Base

```typescript { .api }
interface Message {
  seq: number;
  type: "request" | "response" | "event";
}
```

### Request

```typescript { .api }
interface Request extends Message {
  type: "request";
  command: string;
  arguments?: any;
}
```

### Response

```typescript { .api }
interface Response extends Message {
  type: "response";
  request_seq: number;
  success: boolean;
  command: string;
  message?: string;
  body?: any;
  metadata?: unknown;
  performanceData?: PerformanceData;
}

interface PerformanceData {
  updateGraphDurationMs?: number;
  createAutoImportProviderProgramDurationMs?: number;
  diagnosticsDuration?: FileDiagnosticPerformanceData[];
}
```

### Event

```typescript { .api }
interface Event extends Message {
  type: "event";
  event: string;
  body?: any;
}
```

### Command Types

```typescript { .api }
enum CommandTypes {
  JsxClosingTag = "jsxClosingTag",
  LinkedEditingRange = "linkedEditingRange",
  Brace = "brace",
  BraceCompletion = "braceCompletion",
  GetSpanOfEnclosingComment = "getSpanOfEnclosingComment",
  Change = "change",
  Close = "close",
  Completions = "completions",
  CompletionInfo = "completionInfo",
  CompletionDetails = "completionEntryDetails",
  CompileOnSaveAffectedFileList = "compileOnSaveAffectedFileList",
  CompileOnSaveEmitFile = "compileOnSaveEmitFile",
  Configure = "configure",
  Definition = "definition",
  DefinitionAndBoundSpan = "definitionAndBoundSpan",
  Implementation = "implementation",
  Exit = "exit",
  FileReferences = "fileReferences",
  Format = "format",
  Formatonkey = "formatonkey",
  Geterr = "geterr",
  GeterrForProject = "geterrForProject",
  SemanticDiagnosticsSync = "semanticDiagnosticsSync",
  SyntacticDiagnosticsSync = "syntacticDiagnosticsSync",
  SuggestionDiagnosticsSync = "suggestionDiagnosticsSync",
  NavBar = "navbar",
  Navto = "navto",
  NavTree = "navtree",
  NavTreeFull = "navtree-full",
  DocumentHighlights = "documentHighlights",
  Open = "open",
  Quickinfo = "quickinfo",
  References = "references",
  Reload = "reload",
  Rename = "rename",
  Saveto = "saveto",
  SignatureHelp = "signatureHelp",
  FindSourceDefinition = "findSourceDefinition",
  Status = "status",
  TypeDefinition = "typeDefinition",
  ProjectInfo = "projectInfo",
  ReloadProjects = "reloadProjects",
  Unknown = "unknown",
  OpenExternalProject = "openExternalProject",
  OpenExternalProjects = "openExternalProjects",
  CloseExternalProject = "closeExternalProject",
  UpdateOpen = "updateOpen",
  GetOutliningSpans = "getOutliningSpans",
  TodoComments = "todoComments",
  Indentation = "indentation",
  DocCommentTemplate = "docCommentTemplate",
  CompilerOptionsForInferredProjects = "compilerOptionsForInferredProjects",
  GetCodeFixes = "getCodeFixes",
  GetCombinedCodeFix = "getCombinedCodeFix",
  ApplyCodeActionCommand = "applyCodeActionCommand",
  GetSupportedCodeFixes = "getSupportedCodeFixes",
  GetApplicableRefactors = "getApplicableRefactors",
  GetEditsForRefactor = "getEditsForRefactor",
  GetMoveToRefactoringFileSuggestions = "getMoveToRefactoringFileSuggestions",
  PreparePasteEdits = "preparePasteEdits",
  GetPasteEdits = "getPasteEdits",
  OrganizeImports = "organizeImports",
  GetEditsForFileRename = "getEditsForFileRename",
  ConfigurePlugin = "configurePlugin",
  SelectionRange = "selectionRange",
  ToggleLineComment = "toggleLineComment",
  ToggleMultilineComment = "toggleMultilineComment",
  CommentSelection = "commentSelection",
  UncommentSelection = "uncommentSelection",
  PrepareCallHierarchy = "prepareCallHierarchy",
  ProvideCallHierarchyIncomingCalls = "provideCallHierarchyIncomingCalls",
  ProvideCallHierarchyOutgoingCalls = "provideCallHierarchyOutgoingCalls",
  ProvideInlayHints = "provideInlayHints",
  WatchChange = "watchChange",
  MapCode = "mapCode"
}
```

### File Request Arguments

```typescript { .api }
interface FileRequestArgs {
  file: string;
  projectFileName?: string;
}

interface FileLocationRequestArgs extends FileRequestArgs {
  line: number;
  offset: number;
}

interface FileRangeRequestArgs extends FileRequestArgs {
  startLine: number;
  startOffset: number;
  endLine: number;
  endOffset: number;
}
```

### Common Requests

```typescript { .api }
interface OpenRequest extends Request {
  command: CommandTypes.Open;
  arguments: OpenRequestArgs;
}

interface OpenRequestArgs extends FileRequestArgs {
  fileContent?: string;
  scriptKindName?: ScriptKindName;
  projectRootPath?: string;
}

interface CloseRequest extends Request {
  command: CommandTypes.Close;
  arguments: FileRequestArgs;
}

interface ChangeRequest extends Request {
  command: CommandTypes.Change;
  arguments: ChangeRequestArgs;
}

interface ChangeRequestArgs extends FileRequestArgs {
  line: number;
  offset: number;
  endLine: number;
  endOffset: number;
  insertString: string;
}

interface CompletionsRequest extends Request {
  command: CommandTypes.Completions | CommandTypes.CompletionInfo;
  arguments: CompletionsRequestArgs;
}

interface CompletionsRequestArgs extends FileLocationRequestArgs {
  prefix?: string;
  triggerCharacter?: CompletionsTriggerCharacter;
  triggerKind?: CompletionTriggerKind;
  includeExternalModuleExports?: boolean;
  includeInsertTextCompletions?: boolean;
}

interface DefinitionRequest extends Request {
  command: CommandTypes.Definition;
  arguments: FileLocationRequestArgs;
}

interface ReferencesRequest extends Request {
  command: CommandTypes.References;
  arguments: FileLocationRequestArgs;
}

interface RenameRequest extends Request {
  command: CommandTypes.Rename;
  arguments: RenameRequestArgs;
}

interface RenameRequestArgs extends FileLocationRequestArgs {
  findInStrings?: boolean;
  findInComments?: boolean;
}

interface QuickInfoRequest extends Request {
  command: CommandTypes.Quickinfo;
  arguments: FileLocationRequestArgs;
}

interface SignatureHelpRequest extends Request {
  command: CommandTypes.SignatureHelp;
  arguments: SignatureHelpRequestArgs;
}

interface SignatureHelpRequestArgs extends FileLocationRequestArgs {
  triggerReason?: SignatureHelpTriggerReason;
}

interface CodeFixRequest extends Request {
  command: CommandTypes.GetCodeFixes;
  arguments: CodeFixRequestArgs;
}

interface CodeFixRequestArgs extends FileRangeRequestArgs {
  errorCodes: readonly number[];
}

interface GetApplicableRefactorsRequest extends Request {
  command: CommandTypes.GetApplicableRefactors;
  arguments: GetApplicableRefactorsRequestArgs;
}

interface GetEditsForRefactorRequest extends Request {
  command: CommandTypes.GetEditsForRefactor;
  arguments: GetEditsForRefactorRequestArgs;
}
```

### Common Responses

```typescript { .api }
interface CompletionInfoResponse extends Response {
  body?: CompletionInfo;
}

interface DefinitionResponse extends Response {
  body?: readonly DefinitionInfo[];
}

interface ReferencesResponse extends Response {
  body?: ReferencesResponseBody;
}

interface ReferencesResponseBody {
  refs: readonly ReferencesResponseItem[];
  symbolName: string;
  symbolStartOffset: number;
  symbolDisplayString: string;
}

interface ReferencesResponseItem extends FileSpan {
  lineText: string;
  isWriteAccess: boolean;
  isDefinition: boolean;
}

interface QuickInfoResponse extends Response {
  body?: QuickInfoResponseBody;
}

interface QuickInfoResponseBody {
  kind: ScriptElementKind;
  kindModifiers: string;
  start: Location;
  end: Location;
  displayString: string;
  documentation: string;
  tags: JSDocTagInfo[];
}

interface SignatureHelpResponse extends Response {
  body?: SignatureHelpItems;
}

interface CodeFixResponse extends Response {
  body?: CodeFixAction[];
}

interface GetApplicableRefactorsResponse extends Response {
  body?: ApplicableRefactorInfo[];
}

interface GetEditsForRefactorResponse extends Response {
  body?: RefactorEditInfo;
}
```

### Events

```typescript { .api }
interface DiagnosticEvent extends Event {
  event: "semanticDiag" | "syntaxDiag" | "suggestionDiag";
  body: DiagnosticEventBody;
}

interface DiagnosticEventBody {
  file: string;
  diagnostics: Diagnostic[];
}

interface ConfigFileDiagnosticEvent extends Event {
  event: "configFileDiag";
  body: ConfigFileDiagnosticEventBody;
}

interface ProjectLoadingStartEvent extends Event {
  event: "projectLoadingStart";
  body: ProjectLoadingStartEventBody;
}

interface ProjectLoadingFinishEvent extends Event {
  event: "projectLoadingFinish";
  body: ProjectLoadingFinishEventBody;
}

interface RequestCompletedEvent extends Event {
  event: "requestCompleted";
  body: RequestCompletedEventBody;
}
```

### Location and Span

```typescript { .api }
interface Location {
  line: number;
  offset: number;
}

interface TextSpan {
  start: Location;
  end: Location;
}

interface FileSpan extends TextSpan {
  file: string;
}
```

### Server Classes

```typescript { .api }
namespace ts.server {
  class Session {
    constructor(opts: SessionOptions);
    send(msg: protocol.Message): void;
    event<T extends object>(body: T, eventName: string): void;
    output(info: any, cmdName: string, reqSeq?: number, errorMsg?: string): void;
    executeCommand(request: protocol.Request): HandlerResponse;
  }
  
  class ProjectService {
    readonly typingsInstaller: ITypingsInstaller;
    readonly toCanonicalFileName: (f: string) => string;
    
    openClientFile(fileName: string, fileContent?: string, scriptKind?: ScriptKind, projectRootPath?: string): OpenConfiguredProjectResult;
    closeClientFile(uncheckedFileName: string): void;
    updateProjectGraphs(projects: readonly Project[]): void;
    getDefaultProjectForFile(fileName: NormalizedPath, ensureProject: boolean): Project | undefined;
  }
  
  abstract class Project {
    readonly projectName: string;
    abstract getProjectName(): string;
    abstract getProjectReferences(): readonly ProjectReference[] | undefined;
    abstract hasRoots(): boolean;
    abstract isOrphan(): boolean;
    abstract getLanguageService(ensureSynchronized?: boolean): LanguageService;
    abstract getSourceFile(path: Path): SourceFile | undefined;
    abstract close(): void;
    abstract refreshDiagnostics(): void;
  }
}
```

## Usage Example

Creating a simple TSServer client:

```typescript
import * as ts from 'typescript';
import { protocol } from 'typescript/lib/protocol';

class SimpleClient {
  private seq = 0;
  
  createRequest(command: string, args?: any): protocol.Request {
    return {
      seq: ++this.seq,
      type: "request",
      command,
      arguments: args
    };
  }
  
  openFile(file: string, content: string): protocol.OpenRequest {
    return {
      ...this.createRequest(protocol.CommandTypes.Open),
      arguments: {
        file,
        fileContent: content
      }
    } as protocol.OpenRequest;
  }
  
  getCompletions(file: string, line: number, offset: number): protocol.CompletionsRequest {
    return {
      ...this.createRequest(protocol.CommandTypes.CompletionInfo),
      arguments: {
        file,
        line,
        offset
      }
    } as protocol.CompletionsRequest;
  }
}

const client = new SimpleClient();
const openReq = client.openFile('/path/to/file.ts', 'const x = 42;');
console.log(JSON.stringify(openReq, null, 2));
```
