# Project Agent Rules — Architecture, Refactoring Standards & Granular Commit Discipline
# Context: Project-level rules for d:\kernal

## Code Conventions & Architecture Guidelines

### 1. File Size Limits & Absolute Documentation Preservation
- **Hard Limit:** No source code file (`.py`, `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.sql`, `.toml`, etc.) may exceed **500 lines** (including comments and imports).
- **Proactive Refactor Trigger:** Once a file approaches ~400 lines, proactively evaluate and shatter it rather than letting it hit the ceiling.
- **ZERO REMOVAL OF COMMENTS OR DOCSTRINGS (Strict & Non-Negotiable):**
  - Under NO circumstances may comments, docstrings, architectural rationale, type annotations, or traceability headers be deleted, stripped, truncated, or compressed to save vertical lines.
  - When splitting or relocating code to sibling or child files, all docstrings, parameter specifications, and inline comments MUST be migrated verbatim along with their associated logic.
  - Deleting, omitting, or compressing comments/docstrings to evade the 500-line limit is an instant and severe rule violation.
- **No Line-Gaming:** Never compress readability, merge multiple statements onto one line, or strip valid whitespace to artificially stay under line limits.

### 2. B+ Tree "Folder-as-a-Component" Architecture
Codebases balance and route structure identically to a B+ Tree:

- **Routing Nodes (Index Keys):** Package entry points (`__init__.py`, `mod.rs`, `index.ts`, `lib.rs`) and directory manifests.
  - Sole responsibility: Expose the public API boundary and route calls to internal submodules.
  - Contain zero raw business logic.
  - External consumers must **only** import from this public entry point, never reaching into deep private sub-paths.
- **Leaf Nodes (Data Records):** Implementation files containing focused structs, business logic, workflows, or pure transforms.
- **Structural Invariants:**
  - *Horizontal Split (File > 400 lines):* Split into focused sibling leaf files within the module/domain by technical responsibility:
    - `<feature>.service` / `<feature>.core`: Main orchestration and workflows.
    - `<feature>.types` / `<feature>.models`: Pure schemas, structs, and contracts (zero runtime deps).
    - `<feature>.validator`: Invariant checks, schema guards, and domain rules.
    - `<feature>.utils`: Pure, deterministic helper functions private to the feature.
  - *Vertical Split (Folder > 6–8 files):* Promote the component into a nested sub-directory with its own internal router (`__init__.py` / `mod.rs` / `index.ts`), turning the parent into a higher-level routing node.
  - *No Index Churn:* A folder must represent a genuine domain boundary or cohesive component. Never create a folder containing only a single file.
  - *Strict One-Way Dependency Flow:* Parent folders may import from a child's public index; children may import shared contracts from parents. Siblings must not reach into each other's private sub-files without defined contracts. Zero circular references (`A -> B -> A`).

### 3. Step-by-Step Bottom-Up Refactoring Protocol
When decomposing an overflowing module, refactor strictly bottom-up to prevent import breakage:
1. **Freeze Public Contract First:** Identify every export currently consumed by external callers; ensure these exports remain invariant at the module entry point.
2. **Peel Off Leaf Dependencies First (Bottom-Up):**
   - *Contracts & Types:* Extract schemas, traits, and enums verbatim (with 100% of docstrings intact) to `<feature>.types` (leaf node, zero runtime deps).
   - *Pure Validators & Utilities:* Extract deterministic guards and helpers verbatim (with 100% of docstrings intact) to `<feature>.validator` or `<feature>.utils`.
   - *Core Orchestration:* What remains in the primary file is now compact orchestration logic.
3. **Update the Router:** Re-export the public contract from the folder's entry point (`__init__.py`, `mod.rs`, `index.ts`, `lib.rs`). External consumer import paths remain completely unaffected.
4. **Atomic Verification:** Verify compilation (`py_compile` / `cargo build` / `tsc`) and test suites before committing. Never combine structural file-tree rebalancing with algorithmic changes in the same step.

### 4. Mandatory Granular File-by-File Commit Discipline & Exhaustive Documentation
- **File-by-File Commit Protocol:** Every specific change in every specific file MUST have its own individual, dedicated Git commit. Never bundle or batch changes across multiple files or unrelated modules into a single amorphous commit.
- **Explicit Change Definition:** The commit message header and body must explicitly define the exact structural, type-level, or algorithmic change made in the file.
- **Exhaustive Necessity & Context:** Every commit message MUST document:
  1. *What changed:* Precise, technical description of functions, structs, types, or documentation added, updated, or extracted in the file.
  2. *Why it was necessary (Technical Necessity):* The architectural rationale, bug fix, invariant preservation, or performance requirement necessitating the modification.
  3. *Context & Invariants:* The wider system context, upstream callers/downstream consumers affected, and how compile-time invariants or test suites verify correctness.

### 5. Google Colab Notebook & MCP Execution Standards (Strict Append-Only Invariant)
- **Strict Append-Only Cell Policy:** Under NO circumstances may existing notebook cells or execution outputs in Google Colab be deleted, cleared, overwritten, or modified. Every new test, experiment, verification pass, or benchmark run MUST be appended as a new cell at the end of the notebook.
- **Holistic Historical Visibility:** The full chronological trajectory of notebook cells, source code deployments, outputs, and execution telemetry must remain permanently visible in the session to provide an unbroken overall view of progress and diagnostics.
- **Rigorous Multi-Stack GPU Benchmarking:** All hardware runs on physical GPU silicon (via Google Colab / cloud accelerators) must benchmark and compare implementations across equivalent kernel stacks (e.g., Rust Metaprogrammed Conveyor Belt Sheaf Engine vs. Native CUDA + C++) measuring first-iteration cold-start latency, steady-state compute throughput, memory footprint, and exact numerical parity.

