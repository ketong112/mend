// mend — one patch tool for every AI edit format.
//
// Applies four patch formats to a tree of text files, every file changed or
// none: unified diffs, SEARCH/REPLACE blocks, V4A patches and JSON tool
// calls. See SPEC.md for the contract this file implements.
//
// Usage: node main.ts [--format=NAME] [--check] [--strict] [PATCH]
//   PATCH absent or "-" reads the patch from standard input.
//   --format=NAME reads the patch as that format (auto, unified,
//   search_replace, v4a, json); without it the format is detected.
//   --check reports what would change and writes nothing.
//   --strict matches old lines exactly, with no fuzzy tiers.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from "node:fs";
import { dirname } from "node:path";

type FormatName = "unified" | "search_replace" | "v4a" | "json";

interface Edit {
  run: (tree: FileTree) => void;
}

interface Chunk {
  anchor: string | null;
  oldLines: string[];
  newParts: NewPart[];
  atEnd: boolean;
}

interface NewPart {
  fromFile: boolean;
  oldIndex: number;
  text: string;
}

interface Hunk {
  start: number;
  oldLines: string[];
  newParts: NewPart[];
  added: string[];
  lastAddedNoBreak: boolean;
  lastWrittenNoBreak: boolean;
}

interface Section {
  oldPath: string | null;
  newPath: string | null;
  newFile: boolean;
  deletedFile: boolean;
  hunks: Hunk[];
}

/** The patch is unusable as given: exit 2, `error: usage`. */
class UsageProblem extends Error {}

/** The patch cannot be read: exit 2, `error: no-patch`. */
class NoPatchProblem extends Error {}

/** The patch is malformed at line `line`: exit 2, `error: parse <line>`. */
class ParseProblem extends Error {
  readonly line: number;
  constructor(line: number) {
    super(`parse ${line}`);
    this.line = line;
  }
}

/** One edit failed: exit 1, `error: <code> <path>`. */
class EditProblem extends Error {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    super(`${code} ${path}`);
    this.code = code;
    this.path = path;
  }
}

/** Split patch text into lines: split at \n, one trailing \r removed from
 *  each line, no empty piece after the final \n. */
function patchLines(text: string): string[] {
  const pieces = text.split("\n");
  if (pieces.length > 0 && pieces[pieces.length - 1] === "") pieces.pop();
  return pieces.map((piece) => (piece.endsWith("\r") ? piece.slice(0, -1) : piece));
}

/** A path is safe when it is non-empty, has no leading /, no backslash, and
 *  no empty, `.` or `..` segment. */
function isSafePath(path: string): boolean {
  if (path === "" || path.startsWith("/") || path.includes("\\")) return false;
  for (const part of path.split("/")) {
    if (part === "" || part === "." || part === "..") return false;
  }
  return true;
}

function requireSafe(path: string): void {
  if (!isSafePath(path)) throw new EditProblem("unsafe", path);
}

/** One text file, as lines plus its own line break and whether it ends with
 *  one. The break is CRLF when the file's first break is CR followed by LF,
 *  else LF. */
interface TextFile {
  br: string;
  lines: string[];
  finalBreak: boolean;
}

function splitText(text: string): TextFile {
  const firstNl = text.indexOf("\n");
  const br = firstNl > 0 && text[firstNl - 1] === "\r" ? "\r\n" : "\n";
  const pieces = text.split("\n");
  if (pieces.length > 0 && pieces[pieces.length - 1] === "") pieces.pop();
  const lines = pieces.map((piece) => (piece.endsWith("\r") ? piece.slice(0, -1) : piece));
  return { br, lines, finalBreak: text.endsWith("\n") };
}

function joinText(file: TextFile): string {
  if (file.lines.length === 0) return "";
  return file.lines.join(file.br) + (file.finalBreak ? file.br : "");
}

/** The working tree: original contents read lazily from disk, pending
 *  changes in `now`, successful moves recorded for the output. */
class FileTree {
  private originals = new Map<string, string | null>();
  private now = new Map<string, string | null>();
  private moves: { oldPath: string; newPath: string }[] = [];

  originalOf(path: string): string | null {
    if (!this.originals.has(path)) {
      let value: string | null = null;
      try {
        if (statSync(path).isFile()) value = readFileSync(path, "utf8");
      } catch {
        value = null;
      }
      this.originals.set(path, value);
    }
    return this.originals.get(path)!;
  }

  currentOf(path: string): string | null {
    return this.now.has(path) ? this.now.get(path)! : this.originalOf(path);
  }

  set(path: string, value: string | null): void {
    this.now.set(path, value);
  }

  noteMove(oldPath: string, newPath: string): void {
    this.moves.push({ oldPath, newPath });
  }

  snapshot(): { now: Map<string, string | null>; moves: number } {
    return { now: new Map(this.now), moves: this.moves.length };
  }

  restore(snap: { now: Map<string, string | null>; moves: number }): void {
    this.now = snap.now;
    this.moves.length = snap.moves;
  }

  allPaths(): string[] {
    return [...new Set([...this.originals.keys(), ...this.now.keys()])];
  }

  /** Moves whose effect is still in place: the old path is gone and the new
   *  path holds something. */
  liveMoves(): { oldPath: string; newPath: string }[] {
    return this.moves.filter(
      (m) => this.currentOf(m.oldPath) === null && this.currentOf(m.newPath) !== null,
    );
  }
}

// ---------------------------------------------------------------- matching

function stripTrailing(text: string): string {
  return text.replace(/[ \t]+$/, "");
}

function stripBothEnds(text: string): string {
  return text.replace(/^[ \t]+/, "").replace(/[ \t]+$/, "");
}

function leadingWs(text: string): string {
  const found = /^[ \t]*/.exec(text);
  return found === null ? "" : found[0];
}

function linesEqualAt(a: string, b: string, tier: number): boolean {
  if (tier === 0) return a === b;
  if (tier === 1) return stripTrailing(a) === stripTrailing(b);
  return stripBothEnds(a) === stripBothEnds(b);
}

interface Found {
  tier: number;
  at: number[];
}

/** The first tier (0 exact, 1 trailing whitespace, 2 both ends) at which the
 *  old lines occur at least once, with every occurrence at that tier.
 *  With `strict` only tier 0 is tried. `from` bounds the search below. */
function findOccurrences(
  lines: string[],
  oldLines: string[],
  strict: boolean,
  from: number,
): Found | null {
  const top = strict ? 0 : 2;
  for (let tier = 0; tier <= top; tier += 1) {
    const at: number[] = [];
    for (let i = from; i + oldLines.length <= lines.length; i += 1) {
      let ok = true;
      for (let j = 0; j < oldLines.length; j += 1) {
        if (!linesEqualAt(lines[i + j], oldLines[j], tier)) {
          ok = false;
          break;
        }
      }
      if (ok) at.push(i);
    }
    if (at.length > 0) return { tier, at };
  }
  return null;
}

/** The first line at or after `from` equal to `target` at the first tier
 *  where one is, both sides a single line. */
function findAnchorLine(
  lines: string[],
  target: string,
  strict: boolean,
  from: number,
): number {
  const top = strict ? 0 : 2;
  for (let tier = 0; tier <= top; tier += 1) {
    for (let i = from; i < lines.length; i += 1) {
      if (linesEqualAt(lines[i], target, tier)) return i;
    }
  }
  return -1;
}

/** Build the replacement lines for an occurrence at `pos`: context parts are
 *  written as the file's own matched lines; fresh parts are written as the
 *  patch wrote them, except at tier 2, where a fresh line that is not empty
 *  and not only whitespace and starts with the old lines' leading whitespace
 *  P has that P replaced by the file's first matched line's leading
 *  whitespace F. */
function buildReplacement(
  lines: string[],
  pos: number,
  oldLines: string[],
  newParts: NewPart[],
  tier: number,
): string[] {
  const out: string[] = [];
  for (const part of newParts) {
    if (part.fromFile) {
      out.push(lines[pos + part.oldIndex]);
      continue;
    }
    let text = part.text;
    if (tier === 2 && oldLines.length > 0) {
      const p = leadingWs(oldLines[0]);
      const f = leadingWs(lines[pos]);
      if (p !== f && text !== "" && !/^[ \t]+$/.test(text) && text.startsWith(p)) {
        text = f + text.slice(p.length);
      }
    }
    out.push(text);
  }
  return out;
}

// --------------------------------------------------------------------- CLI

interface Options {
  format: FormatName | null;
  check: boolean;
  strict: boolean;
  patch: string | null;
}

function parseOptions(argv: string[]): Options {
  let format: FormatName | null = null;
  let check = false;
  let strict = false;
  const positional: string[] = [];
  let onlyPositional = false;
  for (const arg of argv) {
    if (!onlyPositional && arg === "--") {
      onlyPositional = true;
      continue;
    }
    if (!onlyPositional && arg.startsWith("--format=")) {
      if (format !== null) throw new UsageProblem();
      const value = arg.slice("--format=".length);
      if (value === "auto") continue;
      if (
        value === "unified" ||
        value === "search_replace" ||
        value === "search-replace" ||
        value === "v4a" ||
        value === "json"
      ) {
        format = value === "search-replace" ? "search_replace" : value;
        continue;
      }
      throw new UsageProblem();
    }
    if (!onlyPositional && arg === "--check") {
      if (check) throw new UsageProblem();
      check = true;
      continue;
    }
    if (!onlyPositional && arg === "--strict") {
      if (strict) throw new UsageProblem();
      strict = true;
      continue;
    }
    if (!onlyPositional && arg.startsWith("-") && arg !== "-") throw new UsageProblem();
    positional.push(arg);
  }
  if (positional.length > 1) throw new UsageProblem();
  return { format, check, strict, patch: positional.length === 0 ? null : positional[0] };
}

function readPatch(patch: string | null): string {
  if (patch === null || patch === "-") return readFileSync(0, "utf8");
  try {
    return readFileSync(patch, "utf8");
  } catch {
    throw new NoPatchProblem();
  }
}

// -------------------------------------------------------------- detection

/** Detect the patch's format from its text, or null when no format fits. */
function detectFormat(text: string): FormatName | null {
  const lines = patchLines(text);
  const first = lines.find((line) => line.trim() !== "");
  if (first !== undefined && first.trim() === "*** Begin Patch") return "v4a";
  if (first !== undefined) {
    const head = first.trimStart()[0];
    if (head === "[" || head === "{") return "json";
  }
  if (lines.some((line) => line.trimEnd() === "<<<<<<< SEARCH")) return "search_replace";
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].startsWith("diff --git ")) return "unified";
    if (
      lines[i].startsWith("--- ") &&
      i + 1 < lines.length &&
      lines[i + 1].startsWith("+++ ")
    ) {
      return "unified";
    }
  }
  return null;
}

// ------------------------------------------------------------------- JSON

function jsonEdit(value: unknown): Edit | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const obj = value as Record<string, unknown>;
  if (typeof obj.path !== "string" || typeof obj.command !== "string") return null;
  const path = obj.path;
  const command = obj.command;

  if (command === "create") {
    if (typeof obj.file_text !== "string") return null;
    const fileText = obj.file_text;
    return {
      run(tree: FileTree): void {
        requireSafe(path);
        if (tree.currentOf(path) !== null) throw new EditProblem("exists", path);
        tree.set(path, fileText);
      },
    };
  }

  if (command === "str_replace") {
    if (
      typeof obj.old_str !== "string" ||
      obj.old_str === "" ||
      (obj.new_str !== undefined && typeof obj.new_str !== "string")
    ) {
      return null;
    }
    const oldStr = obj.old_str;
    const newStr = (obj.new_str as string | undefined) ?? "";
    return {
      run(tree: FileTree): void {
        requireSafe(path);
        const current = tree.currentOf(path);
        if (current === null) throw new EditProblem("missing", path);
        const isCrlf = splitText(current).br === "\r\n";
        const text = isCrlf ? current.replace(/\r\n/g, "\n") : current;
        const oldText = isCrlf ? oldStr.replace(/\r\n/g, "\n") : oldStr;
        const newText = isCrlf ? newStr.replace(/\r\n/g, "\n") : newStr;
        const first = text.indexOf(oldText);
        if (first === -1) throw new EditProblem("no-match", path);
        if (text.indexOf(oldText, first + 1) !== -1) throw new EditProblem("ambiguous", path);
        let out = text.slice(0, first) + newText + text.slice(first + oldText.length);
        if (isCrlf) out = out.replace(/\n/g, "\r\n");
        tree.set(path, out);
      },
    };
  }

  if (command === "insert") {
    if (
      typeof obj.insert_line !== "number" ||
      !Number.isInteger(obj.insert_line) ||
      typeof obj.new_str !== "string"
    ) {
      return null;
    }
    const insertLine = obj.insert_line;
    const newStr = obj.new_str;
    return {
      run(tree: FileTree): void {
        requireSafe(path);
        const current = tree.currentOf(path);
        if (current === null) throw new EditProblem("missing", path);
        const file = splitText(current);
        if (insertLine < 0 || insertLine > file.lines.length) {
          throw new EditProblem("no-match", path);
        }
        file.lines.splice(insertLine, 0, ...patchLines(newStr));
        tree.set(path, joinText(file));
      },
    };
  }

  return null;
}

function parseJson(text: string): Edit[] {
  const lines = patchLines(text);
  const first = lines.findIndex((line) => line.trim() !== "");
  if (first === -1) throw new ParseProblem(1);
  if (lines[first].trimStart().startsWith("[")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new ParseProblem(first + 1);
    }
    if (!Array.isArray(parsed)) throw new ParseProblem(first + 1);
    const edits: Edit[] = [];
    for (const item of parsed) {
      const edit = jsonEdit(item);
      if (edit === null) throw new ParseProblem(first + 1);
      edits.push(edit);
    }
    if (edits.length === 0) throw new ParseProblem(1);
    return edits;
  }
  const edits: Edit[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].trim() === "") continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(lines[i]);
    } catch {
      throw new ParseProblem(i + 1);
    }
    const edit = jsonEdit(parsed);
    if (edit === null) throw new ParseProblem(i + 1);
    edits.push(edit);
  }
  if (edits.length === 0) throw new ParseProblem(1);
  return edits;
}

// --------------------------------------------------------- SEARCH/REPLACE

function makeSearchReplaceEdit(
  path: string,
  oldLines: string[],
  newLines: string[],
  strict: boolean,
): Edit {
  return {
    run(tree: FileTree): void {
      requireSafe(path);
      if (oldLines.length === 0) {
        if (tree.currentOf(path) !== null) throw new EditProblem("exists", path);
        const file: TextFile = {
          br: "\n",
          lines: [...newLines],
          finalBreak: newLines.length > 0,
        };
        tree.set(path, joinText(file));
        return;
      }
      const current = tree.currentOf(path);
      if (current === null) throw new EditProblem("missing", path);
      const file = splitText(current);
      const found = findOccurrences(file.lines, oldLines, strict, 0);
      if (found === null) throw new EditProblem("no-match", path);
      if (found.at.length > 1) throw new EditProblem("ambiguous", path);
      const pos = found.at[0];
      const parts: NewPart[] = newLines.map((text) => ({ fromFile: false, oldIndex: -1, text }));
      const replacement = buildReplacement(file.lines, pos, oldLines, parts, found.tier);
      file.lines.splice(pos, oldLines.length, ...replacement);
      tree.set(path, joinText(file));
    },
  };
}

function parseSearchReplace(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  let previousEnd = -1;
  let previousPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trimEnd() !== "<<<<<<< SEARCH") {
      i += 1;
      continue;
    }
    const searchLine = i + 1;
    let path: string | null = null;
    for (let j = i - 1; j > previousEnd; j -= 1) {
      const trimmed = lines[j].trim();
      if (trimmed === "" || trimmed.startsWith("```")) continue;
      path = trimmed;
      break;
    }
    if (path === null) {
      if (previousPath === null) throw new ParseProblem(searchLine);
      path = previousPath;
    }
    let divider = -1;
    let closer = -1;
    for (let j = i + 1; j < lines.length; j += 1) {
      const trimmed = lines[j].trimEnd();
      if (trimmed === "=======") {
        divider = j;
        break;
      }
      if (trimmed === ">>>>>>> REPLACE") {
        closer = j;
        break;
      }
    }
    if (divider === -1) throw new ParseProblem(searchLine);
    for (let j = divider + 1; j < lines.length; j += 1) {
      if (lines[j].trimEnd() === ">>>>>>> REPLACE") {
        closer = j;
        break;
      }
    }
    if (closer === -1) throw new ParseProblem(searchLine);
    edits.push(
      makeSearchReplaceEdit(path, lines.slice(i + 1, divider), lines.slice(divider + 1, closer), strict),
    );
    previousPath = path;
    previousEnd = closer;
    i = closer + 1;
  }
  if (edits.length === 0) throw new ParseProblem(1);
  return edits;
}

// -------------------------------------------------------------------- V4A

const V4A_PREFIXES = ["*** Add File:", "*** Delete File:", "*** Update File:"] as const;

function v4aSectionKind(line: string): "add" | "delete" | "update" | null {
  for (const prefix of V4A_PREFIXES) {
    if (line.startsWith(prefix) && (line.length === prefix.length || line[prefix.length] === " ")) {
      if (prefix === "*** Add File:") return "add";
      if (prefix === "*** Delete File:") return "delete";
      return "update";
    }
  }
  return null;
}

function parseV4aChunks(
  raw: { text: string; num: number }[],
): Chunk[] {
  const chunks: Chunk[] = [];
  let current: Chunk | null = null;
  for (const entry of raw) {
    const text = entry.text;
    if (text.startsWith("@@")) {
      const anchor = text.slice(2).trim();
      current = { anchor: anchor === "" ? null : anchor, oldLines: [], newParts: [], atEnd: false };
      chunks.push(current);
      continue;
    }
    if (text.trimEnd() === "*** End of File") {
      if (current === null) throw new ParseProblem(entry.num);
      current.atEnd = true;
      current = null;
      continue;
    }
    if (text === "" || text.startsWith(" ") || text.startsWith("-") || text.startsWith("+")) {
      if (current === null) {
        current = { anchor: null, oldLines: [], newParts: [], atEnd: false };
        chunks.push(current);
      }
      if (text === "") {
        current.oldLines.push("");
        current.newParts.push({ fromFile: true, oldIndex: current.oldLines.length - 1, text: "" });
      } else if (text.startsWith(" ")) {
        const body = text.slice(1);
        current.oldLines.push(body);
        current.newParts.push({ fromFile: true, oldIndex: current.oldLines.length - 1, text: "" });
      } else if (text.startsWith("-")) {
        current.oldLines.push(text.slice(1));
      } else {
        current.newParts.push({ fromFile: false, oldIndex: -1, text: text.slice(1) });
      }
      continue;
    }
    throw new ParseProblem(entry.num);
  }
  return chunks;
}

function makeV4aUpdateEdit(
  path: string,
  moveTo: string | null,
  chunks: Chunk[],
  strict: boolean,
): Edit {
  return {
    run(tree: FileTree): void {
      requireSafe(path);
      if (moveTo !== null) requireSafe(moveTo);
      const current = tree.currentOf(path);
      if (current === null) throw new EditProblem("missing", path);
      if (moveTo !== null && tree.currentOf(moveTo) !== null) {
        throw new EditProblem("exists", moveTo);
      }
      const file = splitText(current);
      let cursor = 0;
      for (const chunk of chunks) {
        if (chunk.anchor !== null) {
          const at = findAnchorLine(file.lines, chunk.anchor, strict, cursor);
          if (at === -1) throw new EditProblem("no-match", path);
          cursor = at + 1;
        }
        if (chunk.oldLines.length === 0) {
          const at = chunk.atEnd ? file.lines.length : cursor;
          const replacement = buildReplacement(file.lines, at, [], chunk.newParts, -1);
          file.lines.splice(at, 0, ...replacement);
          cursor = at + replacement.length;
          continue;
        }
        const found = findOccurrences(file.lines, chunk.oldLines, strict, cursor);
        if (found === null) throw new EditProblem("no-match", path);
        let pos = found.at[0];
        if (chunk.atEnd) {
          const ending = found.at.filter((p) => p + chunk.oldLines.length === file.lines.length);
          if (ending.length === 0) throw new EditProblem("no-match", path);
          pos = ending[0];
        }
        const replacement = buildReplacement(file.lines, pos, chunk.oldLines, chunk.newParts, found.tier);
        file.lines.splice(pos, chunk.oldLines.length, ...replacement);
        cursor = pos + replacement.length;
      }
      const out = joinText(file);
      if (moveTo !== null) {
        tree.set(moveTo, out);
        tree.set(path, null);
        tree.noteMove(path, moveTo);
      } else {
        tree.set(path, out);
      }
    },
  };
}

function parseV4a(lines: string[], strict: boolean): Edit[] {
  let first = -1;
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].trim() !== "") {
      first = i;
      break;
    }
  }
  if (first === -1) throw new ParseProblem(1);
  if (lines[first].trim() !== "*** Begin Patch") throw new ParseProblem(first + 1);
  let last = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i].trim() !== "") {
      last = i;
      break;
    }
  }
  if (lines[last].trim() !== "*** End Patch") throw new ParseProblem(lines.length + 1);

  const edits: Edit[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i];
    if (line.trim() === "") {
      i += 1;
      continue;
    }
    const kind = v4aSectionKind(line);
    if (kind === null) throw new ParseProblem(i + 1);
    const prefix = V4A_PREFIXES[kind === "add" ? 0 : kind === "delete" ? 1 : 2];
    const path = line.slice(prefix.length).trim();
    if (kind === "add") {
      const raw: { text: string; num: number }[] = [];
      let j = i + 1;
      while (j < last && v4aSectionKind(lines[j]) === null) {
        raw.push({ text: lines[j], num: j + 1 });
        j += 1;
      }
      while (raw.length > 0 && raw[raw.length - 1].text.trim() === "") raw.pop();
      const fileLines: string[] = [];
      for (const entry of raw) {
        if (entry.text === "") fileLines.push("");
        else if (entry.text.startsWith("+")) fileLines.push(entry.text.slice(1));
        else throw new ParseProblem(entry.num);
      }
      edits.push({
        run(tree: FileTree): void {
          requireSafe(path);
          if (tree.currentOf(path) !== null) throw new EditProblem("exists", path);
          const file: TextFile = {
            br: "\n",
            lines: [...fileLines],
            finalBreak: fileLines.length > 0,
          };
          tree.set(path, joinText(file));
        },
      });
      i = j;
      continue;
    }
    if (kind === "delete") {
      edits.push({
        run(tree: FileTree): void {
          requireSafe(path);
          if (tree.currentOf(path) === null) throw new EditProblem("missing", path);
          tree.set(path, null);
        },
      });
      i += 1;
      continue;
    }
    let moveTo: string | null = null;
    let j = i + 1;
    if (j < last) {
      const moveLine = lines[j];
      if (moveLine.startsWith("*** Move to:") && (moveLine.length === 12 || moveLine[12] === " ")) {
        moveTo = moveLine.slice(12).trim();
        j += 1;
      }
    }
    const raw: { text: string; num: number }[] = [];
    while (j < last && v4aSectionKind(lines[j]) === null) {
      raw.push({ text: lines[j], num: j + 1 });
      j += 1;
    }
    while (raw.length > 0 && raw[raw.length - 1].text.trim() === "") raw.pop();
    const chunks = parseV4aChunks(raw);
    if (chunks.length === 0 && moveTo === null) throw new ParseProblem(i + 1);
    edits.push(makeV4aUpdateEdit(path, moveTo, chunks, strict));
    i = j;
  }
  if (edits.length === 0) throw new ParseProblem(1);
  return edits;
}

// ---------------------------------------------------------------- unified

function stripPathPrefix(path: string): string {
  if (path.startsWith("a/") || path.startsWith("b/")) return path.slice(2);
  return path;
}

function headerPath(line: string): string {
  let path = line.slice(4);
  const tab = path.indexOf("\t");
  if (tab !== -1) path = path.slice(0, tab);
  return path.trim();
}

function namedPath(path: string): string | null {
  return path === "/dev/null" ? null : stripPathPrefix(path);
}

function parseDiffGit(line: string, num: number): [string, string] {
  const rest = line.slice("diff --git ".length).trim();
  const parts = rest.split(/\s+/);
  if (parts.length !== 2 || !parts[0].startsWith("a/") || !parts[1].startsWith("b/")) {
    throw new ParseProblem(num);
  }
  return [parts[0].slice(2), parts[1].slice(2)];
}

function parseHunkHeader(line: string): number | null {
  const found = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
  if (found === null) return null;
  return parseInt(found[1], 10);
}

function finishHunk(start: number, raw: { text: string; num: number }[]): Hunk {
  const oldLines: string[] = [];
  const newParts: NewPart[] = [];
  const added: string[] = [];
  let lastAdded = -1;
  let lastWritten = -1;
  for (let k = 0; k < raw.length; k += 1) {
    const text = raw[k].text;
    if (text === "") {
      oldLines.push("");
      newParts.push({ fromFile: true, oldIndex: oldLines.length - 1, text: "" });
      lastWritten = k;
    } else if (text.startsWith(" ")) {
      const body = text.slice(1);
      oldLines.push(body);
      newParts.push({ fromFile: true, oldIndex: oldLines.length - 1, text: "" });
      lastWritten = k;
    } else if (text.startsWith("-")) {
      oldLines.push(text.slice(1));
    } else if (text.startsWith("+")) {
      const body = text.slice(1);
      newParts.push({ fromFile: false, oldIndex: -1, text: body });
      added.push(body);
      lastAdded = k;
      lastWritten = k;
    }
    // A "\" line refers to the body line before it; it adds no line.
  }
  const followsBackslash = (index: number): boolean =>
    index >= 0 && index + 1 < raw.length && raw[index + 1].text.startsWith("\\");
  return {
    start,
    oldLines,
    newParts,
    added,
    lastAddedNoBreak: followsBackslash(lastAdded),
    lastWrittenNoBreak: followsBackslash(lastWritten),
  };
}

function parseUnified(lines: string[]): Section[] {
  const sections: Section[] = [];
  let current: Section | null = null;
  let state: "out" | "header" | "between" = "out";
  let i = 0;

  const openSection = (oldPath: string | null, newPath: string | null): void => {
    if (current !== null) sections.push(current);
    current = { oldPath, newPath, newFile: false, deletedFile: false, hunks: [] };
    state = "header";
  };
  const closeSection = (): void => {
    if (current !== null) sections.push(current);
    current = null;
    state = "out";
  };
  const collectHunk = (start: number): void => {
    i += 1;
    const raw: { text: string; num: number }[] = [];
    while (i < lines.length) {
      const body = lines[i];
      if (body.startsWith("diff --git ")) break;
      if (
        body.startsWith("--- ") &&
        i + 1 < lines.length &&
        lines[i + 1].startsWith("+++ ")
      ) {
        break;
      }
      if (body.startsWith("@@")) break;
      if (
        body === "" ||
        body.startsWith(" ") ||
        body.startsWith("-") ||
        body.startsWith("+") ||
        body.startsWith("\\")
      ) {
        raw.push({ text: body, num: i + 1 });
        i += 1;
        continue;
      }
      break;
    }
    while (raw.length > 0 && raw[raw.length - 1].text === "") raw.pop();
    current!.hunks.push(finishHunk(start, raw));
    state = "between";
  };

  while (i < lines.length) {
    const line = lines[i];
    const num = i + 1;
    if (state === "out") {
      if (line.startsWith("diff --git ")) {
        const [oldPath, newPath] = parseDiffGit(line, num);
        openSection(oldPath, newPath);
        i += 1;
        continue;
      }
      if (
        line.startsWith("--- ") &&
        i + 1 < lines.length &&
        lines[i + 1].startsWith("+++ ")
      ) {
        openSection(namedPath(headerPath(line)), namedPath(headerPath(lines[i + 1])));
        i += 2;
        continue;
      }
      if (line.startsWith("@@")) throw new ParseProblem(num);
      i += 1;
      continue;
    }
    if (state === "header") {
      if (line.startsWith("diff --git ")) {
        closeSection();
        continue;
      }
      if (line.startsWith("--- ")) {
        if (i + 1 < lines.length && lines[i + 1].startsWith("+++ ")) {
          current!.oldPath = namedPath(headerPath(line));
          current!.newPath = namedPath(headerPath(lines[i + 1]));
          i += 2;
          continue;
        }
        throw new ParseProblem(num + 1);
      }
      if (line.startsWith("rename from ")) {
        current!.oldPath = namedPath(line.slice("rename from ".length).trim());
        i += 1;
        continue;
      }
      if (line.startsWith("rename to ")) {
        current!.newPath = namedPath(line.slice("rename to ".length).trim());
        i += 1;
        continue;
      }
      if (line.startsWith("new file mode")) {
        current!.newFile = true;
        i += 1;
        continue;
      }
      if (line.startsWith("deleted file mode")) {
        current!.deletedFile = true;
        i += 1;
        continue;
      }
      const start = parseHunkHeader(line);
      if (start !== null) {
        collectHunk(start);
        continue;
      }
      if (line.startsWith("@@")) throw new ParseProblem(num);
      i += 1;
      continue;
    }
    // state === "between": after a hunk body, still in the section.
    if (line.startsWith("diff --git ")) {
      closeSection();
      continue;
    }
    if (
      line.startsWith("--- ") &&
      i + 1 < lines.length &&
      lines[i + 1].startsWith("+++ ")
    ) {
      closeSection();
      continue;
    }
    const start = parseHunkHeader(line);
    if (start !== null) {
      collectHunk(start);
      continue;
    }
    if (line.startsWith("@@")) throw new ParseProblem(num);
    i += 1;
  }
  if (current !== null) sections.push(current);
  if (sections.length === 0) throw new ParseProblem(1);
  return sections;
}

// ------------------------------------------------------- applying hunks

function applyHunk(file: TextFile, hunk: Hunk, strict: boolean, delta: number, path: string): TextFile {
  if (hunk.oldLines.length === 0) {
    const at = Math.min(Math.max(hunk.start + delta, 0), file.lines.length);
    const replacement = buildReplacement(file.lines, at, [], hunk.newParts, -1);
    const lines = [...file.lines.slice(0, at), ...replacement, ...file.lines.slice(at)];
    let finalBreak = file.finalBreak;
    if (at === file.lines.length) finalBreak = !hunk.lastWrittenNoBreak;
    return { br: file.br, lines, finalBreak };
  }
  const found = findOccurrences(file.lines, hunk.oldLines, strict, 0);
  if (found === null) throw new EditProblem("no-match", path);
  const target = hunk.start - 1 + delta;
  let pos = found.at[0];
  let best = Math.abs(pos - target);
  for (const candidate of found.at) {
    const distance = Math.abs(candidate - target);
    if (distance < best) {
      pos = candidate;
      best = distance;
    }
  }
  const replacement = buildReplacement(file.lines, pos, hunk.oldLines, hunk.newParts, found.tier);
  const lines = [
    ...file.lines.slice(0, pos),
    ...replacement,
    ...file.lines.slice(pos + hunk.oldLines.length),
  ];
  let finalBreak = file.finalBreak;
  if (pos + hunk.oldLines.length === file.lines.length) {
    finalBreak = !hunk.lastWrittenNoBreak;
  }
  return { br: file.br, lines, finalBreak };
}

function makeUnifiedEdit(section: Section, strict: boolean): Edit {
  return {
    run(tree: FileTree): void {
      if (section.oldPath !== null) requireSafe(section.oldPath);
      if (section.newPath !== null) requireSafe(section.newPath);
      const isCreate = section.oldPath === null || section.newFile;
      if (isCreate) {
        const newPath = section.newPath;
        if (newPath === null) return;
        if (tree.currentOf(newPath) !== null) throw new EditProblem("exists", newPath);
        const lines: string[] = [];
        let finalBreak = false;
        for (const hunk of section.hunks) {
          if (hunk.added.length > 0) {
            lines.push(...hunk.added);
            finalBreak = !hunk.lastAddedNoBreak;
          }
        }
        const file: TextFile = { br: "\n", lines, finalBreak: lines.length > 0 && finalBreak };
        tree.set(newPath, joinText(file));
        return;
      }
      const isDelete = section.newPath === null || section.deletedFile;
      if (isDelete) {
        const oldPath = section.oldPath!;
        if (tree.currentOf(oldPath) === null) throw new EditProblem("missing", oldPath);
        tree.set(oldPath, null);
        return;
      }
      const oldPath = section.oldPath!;
      const current = tree.currentOf(oldPath);
      if (current === null) throw new EditProblem("missing", oldPath);
      const isMove = section.newPath !== null && section.newPath !== oldPath;
      if (isMove && tree.currentOf(section.newPath!) !== null) {
        throw new EditProblem("exists", section.newPath!);
      }
      let file = splitText(current);
      let delta = 0;
      for (const hunk of section.hunks) {
        file = applyHunk(file, hunk, strict, delta, oldPath);
        delta += hunk.newParts.length - hunk.oldLines.length;
      }
      const out = joinText(file);
      if (isMove) {
        tree.set(section.newPath!, out);
        tree.set(oldPath, null);
        tree.noteMove(oldPath, section.newPath!);
      } else {
        tree.set(oldPath, out);
      }
    },
  };
}

// ------------------------------------------------------------- running

function parsePatch(format: FormatName, text: string, strict: boolean): Edit[] {
  const lines = patchLines(text);
  if (format === "unified") {
    return parseUnified(lines).map((section) => makeUnifiedEdit(section, strict));
  }
  if (format === "search_replace") return parseSearchReplace(lines, strict);
  if (format === "v4a") return parseV4a(lines, strict);
  return parseJson(text);
}

function compareUtf8(a: string, b: string): number {
  return Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

interface RunOutcome {
  code: number;
  out: string[];
}

function run(argv: string[]): RunOutcome {
  const options = parseOptions(argv);
  const text = readPatch(options.patch);
  const format = options.format ?? detectFormat(text);
  if (format === null) return { code: 2, out: ["error: format"] };
  const edits = parsePatch(format, text, options.strict);
  const tree = new FileTree();
  const failures: EditProblem[] = [];
  for (const edit of edits) {
    const snap = tree.snapshot();
    try {
      edit.run(tree);
    } catch (error) {
      if (error instanceof EditProblem) {
        tree.restore(snap);
        failures.push(error);
        continue;
      }
      throw error;
    }
  }
  if (failures.length > 0) {
    return {
      code: 1,
      out: [
        ...failures.map((f) => `error: ${f.code} ${f.path}`),
        `failed: ${failures.length} of ${edits.length} edits`,
      ],
    };
  }
  const rows: { key: string; line: string }[] = [];
  const movedOld = new Set<string>();
  const movedNew = new Set<string>();
  for (const move of tree.liveMoves()) {
    rows.push({ key: move.newPath, line: `R ${move.oldPath} -> ${move.newPath}` });
    movedOld.add(move.oldPath);
    movedNew.add(move.newPath);
  }
  for (const path of tree.allPaths()) {
    if (movedOld.has(path) || movedNew.has(path)) continue;
    const was = tree.originalOf(path);
    const now = tree.currentOf(path);
    if (was === now) continue;
    rows.push({ key: path, line: `${was === null ? "A" : now === null ? "D" : "M"} ${path}` });
  }
  rows.sort((a, b) => compareUtf8(a.key, b.key));
  if (!options.check) {
    for (const path of tree.allPaths()) {
      const was = tree.originalOf(path);
      const now = tree.currentOf(path);
      if (was === now) continue;
      if (now === null) rmSync(path, { force: true });
      else {
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, now);
      }
    }
  }
  const count = rows.length;
  return {
    code: 0,
    out: [...rows.map((row) => row.line), `ok: ${count} ${count === 1 ? "file" : "files"}`],
  };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== "") {
  // A trial: the kit's runner reads the checks from the env and runs this
  // file on each one, without this variable.
  const { runTrial } = await import("./runner.ts");
  await runTrial(trialEnv);
} else {
  try {
    const result = run(process.argv.slice(2));
    writeSync(1, result.out.map((line) => `${line}\n`).join(""));
    process.exitCode = result.code;
  } catch (error) {
    if (error instanceof UsageProblem) {
      writeSync(1, "error: usage\n");
      process.exitCode = 2;
    } else if (error instanceof NoPatchProblem) {
      writeSync(1, "error: no-patch\n");
      process.exitCode = 2;
    } else if (error instanceof ParseProblem) {
      writeSync(1, `error: parse ${error.line}\n`);
      process.exitCode = 2;
    } else {
      throw error;
    }
  }
}
