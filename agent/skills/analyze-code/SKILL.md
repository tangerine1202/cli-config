---
name: analyze-code
description: Analyzes code quality for entire files or git changes. Identifies common issues, anti-patterns, and improvement opportunities. ALWAYS produces an implementation_plan.md. Use when the user asks to "analyze this code", "review my git changes", "check code quality", or "review this".
allowed-tools: Read, Glob, Grep, WebSearch, Bash(git:*)
---

# Analyze Code Skill

Fast code quality analysis that identifies issues and provides actionable recommendations in a concise format. It adapts its scope based on the user's request: it can analyze entire files or just the recent git changes (staged, unstaged, or recent commits).

## When to Use This Skill

Trigger when users ask to:
- "Analyze this code" or "Check code quality"
- "Review my git changes" or "Check what I changed"
- Identify anti-patterns
- Review before committing

## Analysis Methodology

### 1. Determine Scope
If the user asks to analyze specific files:
- Use the Read tool to examine the target files.
If the user asks to review git changes:
- Run `git status` and `git diff` / `git diff --cached` / `git diff --name-only HEAD`.
- Identify the changed files and the specific changed lines.

### 2. Understand the Code
- Identify main components and responsibilities.
- Note the technology stack and dependencies.
- Map out data flow.

### 3. Identify Issues
Look for:
- **Code Structure / Anti-Patterns**: Hardcoded values, duplication, god objects/functions, deep nesting, missing types.
- **Design Principles (SOLID)**: Single responsibility, DRY, etc.
- **Performance**: Algorithmic complexity, unnecessary iterations.
- **Maintainability**: Poor naming, missing error handling, tight coupling.
- **Security**: Injection vulnerabilities, missing validation, hardcoded secrets.

When reviewing git changes, pay special attention to:
- New duplication or complexity.
- Appropriate test coverage for the new changes.
- Unintended behavior.

### 4. Create implementation_plan.md (REQUIRED)
You MUST ALWAYS output the results of this analysis to a file named `implementation_plan.md`. This standardizes the handoff to the `refactor-code` skill.

Format the plan as follows:

```markdown
# Code Quality Analysis Plan

**Scope:** [Files Analyzed or Git Changes Analyzed]

## Summary
[Brief summary of overall quality and findings]

## Critical Issues (Must Fix)
1. **[Issue Name]** - [Location]
   - **Problem:** [Description]
   - **Proposed Fix:** [Actionable steps or code snippet]

## Moderate Issues (Should Fix)
[List of moderate issues]

## Minor Issues (Nice to Have)
[List of minor issues]
```

### 5. Present Summary to User
After saving the plan, present a brief summary in the chat:
- Number of issues found (Critical/Moderate/Minor).
- The top 1-2 critical issues.
- State that the full plan is in `implementation_plan.md`.
- Suggest they run the `code-review` workflow or the `refactor-code` skill to apply the fixes.
