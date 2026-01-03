---
name: critic
version: 1.0.0
description: Comprehensive code quality workflow that analyzes code for issues, creates improvement plan, and optionally applies refactoring. Use when the user asks to "critic my code", "review and improve this", "ensure code quality", "analyze and fix", or wants a complete quality audit with optional improvements.
---

# Code Critic Skill

This skill provides a complete code quality assurance workflow by orchestrating the `code-analysis` and `refactor-code` skills. It analyzes code for weaknesses, creates a detailed improvement plan, and optionally applies the refactoring changes.

## When to Use This Skill

Trigger this skill when users ask to:
- "Critic my code" or "be critical of this code"
- "Review and improve this code"
- "Ensure code quality"
- "Analyze and fix issues"
- "Give me a complete code audit"
- "Check this code and make it better"
- "What's wrong with this code and how do I fix it?"

**This is a convenience skill** that combines analysis + refactoring into one workflow. Users can also invoke `code-analysis` and `refactor-code` separately if they prefer.

## Workflow Overview

```
1. Code Analysis Phase
   ├─ Read and analyze code
   ├─ Identify issues (critical → moderate → minor)
   ├─ Search for best practices
   └─ Create implementation_plan.md

2. Review & Decision Point
   ├─ Present findings summary
   ├─ Highlight critical issues
   └─ Ask user: Proceed with refactoring? [Yes/No/Selective]

3. Refactoring Phase (if approved)
   ├─ Read implementation_plan.md
   ├─ Apply changes iteratively
   ├─ Verify each change with tests
   └─ Report final results
```

## Step-by-Step Process

### Phase 1: Analysis

**1. Understand the request:**
- Identify which files/components the user wants critiqued
- Clarify scope if ambiguous (single file vs entire module vs codebase)
- Ask user to specify areas of concern if not mentioned (performance, security, maintainability, etc.)

**2. Run code-analysis workflow:**

Follow the methodology from the `code-analysis` skill:

a. **Read the code:**
   - Use Read tool for specified files
   - Understand architecture and data flows
   - Map out key components

b. **Identify weaknesses:**
   - Look for anti-patterns (hardcoded values, duplication, tight coupling)
   - Check SOLID principles
   - Analyze scalability, performance, maintainability

c. **Challenge critically:**
   - Scalability: "Will this work at 10x scale?"
   - Performance: "Are there obvious bottlenecks?"
   - Idiomatic: "Does this follow language/framework conventions?"
   - Security: "Are there vulnerabilities?"

d. **Search for best practices:**
   - Use Grep to find internal patterns in codebase
   - Use WebSearch for external best practices
   - Compare current code against standards

e. **Create implementation_plan.md:**
   - Categorize issues: Critical / Moderate / Minor
   - Provide specific solutions with code examples
   - Create prioritized roadmap
   - Include testing strategy

**3. Present analysis summary to user:**

```markdown
## Code Critique Summary

### Files Analyzed
- [List of files with line counts]

### Issue Severity Breakdown
- 🔴 Critical Issues: X found (must fix)
- 🟡 Moderate Issues: Y found (should fix)
- 🟢 Minor Issues: Z found (nice to have)

### Top 3 Critical Issues
1. [Issue name] - [Brief impact statement]
2. [Issue name] - [Brief impact statement]
3. [Issue name] - [Brief impact statement]

### Detailed Analysis
Full analysis saved to: `implementation_plan.md`

You can:
1. Review the detailed plan
2. Ask questions about specific findings
3. Proceed with automated refactoring (I can apply the fixes)
4. Apply fixes manually using the plan as a guide
```

### Phase 2: Decision Point

**Ask user for direction using AskUserQuestion:**

Present 3 options:
1. **Apply All Fixes** - Automatically refactor all issues (Critical → Moderate → Minor)
2. **Apply Critical Only** - Fix only critical issues automatically
3. **Manual Review** - User will apply fixes manually using the plan

**Include context in the question:**
- Number of files to be modified
- Estimated scope (Small: <5 changes, Medium: 5-15 changes, Large: >15 changes)
- Risk level (Low/Medium/High based on test coverage and complexity)

**Example question format:**
```
I found 12 issues across 4 files (3 critical, 5 moderate, 4 minor).
Estimated refactoring scope: Medium
Risk level: Low (good test coverage detected)

How would you like to proceed?
```

### Phase 3: Refactoring (Conditional)

**If user chooses automated refactoring:**

Follow the methodology from the `refactor-code` skill:

**1. Prepare:**
- Read `implementation_plan.md`
- Read all files to be modified
- Identify existing tests
- Create TodoWrite list for tracking progress

**2. Apply changes iteratively:**

For each issue (in priority order):

a. **Make focused change:**
   - Modify only what's needed for THIS issue
   - Use Edit tool for precise changes
   - Keep changes atomic and focused

b. **Verify immediately:**
   - Run relevant tests after EACH change
   - Check for lint/format errors
   - Validate no regressions

c. **Fix if broken:**
   - If tests fail, fix immediately before proceeding
   - Don't accumulate broken changes

d. **Track progress:**
   - Mark todo as completed
   - Move to next issue

**3. Handle different user choices:**

**If "Apply All Fixes":**
- Process all issues: Critical → Moderate → Minor
- Stop and ask if any step fails repeatedly

**If "Apply Critical Only":**
- Process only critical issues from implementation_plan.md
- Note moderate/minor issues remain for later

**If "Manual Review":**
- Skip refactoring phase entirely
- Remind user that implementation_plan.md contains the roadmap

**4. Final verification:**
- Run full test suite
- Run linters/formatters
- Report summary of changes

### Phase 4: Summary Report

**Provide comprehensive summary:**

```markdown
## Code Critique Complete

### Analysis Phase Results
- Files analyzed: X
- Issues found: Y total (A critical, B moderate, C minor)
- Implementation plan: `implementation_plan.md`

### Refactoring Phase Results
[If refactoring was performed]
- Changes applied: X modifications across Y files
- Tests: All passing ✅ (or details if failures)
- Files modified:
  - `file1.py`: [Description of changes]
  - `file2.py`: [Description of changes]

[If refactoring was skipped]
- No changes applied (user chose manual review)
- Use `implementation_plan.md` as your guide

### Remaining Items
[If any issues remain]
- X moderate issues not addressed
- Y minor issues not addressed
- See `implementation_plan.md` for details

### Recommendations
[Any follow-up suggestions]
```

## Important Guidelines

**Scope Management:**
- Start with explicit user request
- Don't expand scope without asking
- If user says "critic my code" without specifics, ask which files/components
- Respect time constraints - suggest phased approach for large codebases

**Communication:**
- Be honest about issues found (this is a critique!)
- Balance criticism with constructive solutions
- Quantify impact where possible ("5x slower" vs "not optimal")
- Celebrate what's done well too

**Safety:**
- Always create implementation_plan.md before any refactoring
- Never apply changes without user approval
- Make changes incrementally with verification
- Stop immediately if tests fail repeatedly

**Transparency:**
- Show what you're doing at each phase
- Explain why issues are critical vs moderate
- Provide evidence for claims (line numbers, examples)
- Link to best practices and documentation

**Flexibility:**
- Allow user to abort at any decision point
- Support partial refactoring (critical only)
- Respect user's preference for manual vs automated fixes

## Integration with Other Skills

**Uses these skills internally:**
1. `code-analysis` - Analysis methodology and output format
2. `refactor-code` - Refactoring methodology and patterns

**This skill does NOT call those skills directly** - it follows their documented methodologies to provide a unified workflow.

**Users can also:**
- Run `code-analysis` alone for analysis-only
- Run `refactor-code` alone if they already have a plan
- Use `critic` for the full end-to-end workflow

## Example Usage

### Example 1: Single File Review

**User:** "Critic my factory_env.py file"

**Skill Actions:**
1. Read `factory_env.py` (1265 lines)
2. Analyze for issues:
   - Find: 3 critical (performance bottlenecks), 5 moderate (code duplication), 2 minor (naming)
3. Search codebase for patterns (how similar code is handled elsewhere)
4. Search web for "Isaac Lab performance best practices"
5. Create `implementation_plan.md`
6. Present summary: "Found 10 issues, 3 critical performance bottlenecks"
7. Ask: "Apply all fixes, critical only, or manual review?"
8. [User chooses "Critical only"]
9. Apply 3 critical fixes with tests after each
10. Report: "Fixed 3 critical issues, 7 remain in plan for later"

### Example 2: Module Review (Multiple Files)

**User:** "Review and improve the factory task module for code quality"

**Skill Actions:**
1. Clarify scope: "The factory module has 8 files. Should I review all of them or specific ones?"
2. [User: "Focus on factory_env.py, factory_control.py, and curriculum.py"]
3. Read all 3 files
4. Analyze across files:
   - Find: 2 critical (security issue, scalability issue), 8 moderate, 5 minor
   - Identify cross-file duplication
5. Create comprehensive `implementation_plan.md`
6. Present summary: "Found 15 issues across 3 files"
7. Ask: "This is a medium-scope refactoring (12 changes). Apply all, critical only, or manual?"
8. [User: "Apply all fixes"]
9. Create todo list with 15 items
10. Apply changes iteratively with verification
11. Report: "Refactored 3 files, all tests passing, 15 improvements applied"

### Example 3: Security-Focused Review

**User:** "Critic this authentication code for security issues"

**Skill Actions:**
1. Read authentication files
2. Focus analysis on security:
   - Check for OWASP Top 10 vulnerabilities
   - Look for hardcoded secrets
   - Verify input validation
   - Check password handling
3. Search "OWASP authentication best practices 2025"
4. Create `implementation_plan.md` emphasizing security issues
5. Present: "Found 2 CRITICAL security vulnerabilities (hardcoded secret, SQL injection risk)"
6. Recommend: "Suggest applying critical security fixes immediately"
7. [User: "Yes, fix now"]
8. Apply security fixes
9. Verify with security-specific tests
10. Report: "Fixed 2 critical security issues, recommend penetration testing"

## Quick Reference

**Invocation patterns:**
- "critic [file/module/component]"
- "review and improve [code]"
- "ensure code quality of [target]"
- "what's wrong with this code and fix it"

**Phases:**
1. **Analysis** (always runs) → Creates implementation_plan.md
2. **Decision** (always asks) → User chooses next step
3. **Refactoring** (conditional) → Applies fixes if approved
4. **Summary** (always runs) → Reports results

**Key files:**
- Input: User-specified code files
- Output: `implementation_plan.md` (analysis results)
- Modified: Files being refactored (if Phase 3 runs)

**Dependencies:**
- Methodology from `code-analysis` skill
- Methodology from `refactor-code` skill
- Tools: Read, Grep, WebSearch (analysis), Edit, Write, Bash (refactoring)
