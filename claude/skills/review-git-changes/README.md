# Review Git Changes Skill

A focused code quality skill that analyzes and refactors only the code you've changed in git, ensuring your modifications follow best practices before committing.

## Quick Start

```bash
# Make some code changes
vim src/myfile.py

# Review your changes
"Review my git changes"

# Or be more specific
"Review my changes before commit"
"Check what I changed and make sure it's clean"
"Ensure my changes follow best practices"
```

## What Makes This Skill Different?

| Skill | Scope | Use Case |
|-------|-------|----------|
| **review-git-changes** | Only git diff (changed code) | Pre-commit review, ensuring changes are clean |
| **code-analysis** | Entire files/modules | Full file quality assessment |
| **refactor-code** | Based on implementation plan | Systematic refactoring from analysis |
| **critic** | Full audit + optional refactoring | Complete code quality workflow |

**Key Advantage:** By focusing only on what you changed, this skill:
- Is much faster than full file analysis
- Catches issues in your changes before they enter git history
- Doesn't overwhelm you with issues in existing code
- Provides targeted, actionable feedback

## Core Principles

### 1. Safety First
- Never skip tests
- Make changes incrementally (one at a time)
- Run verification after EACH change
- Keep git state clean (easy to rollback)

### 2. Maintain Functionality
- Refactoring ≠ changing behavior
- All existing tests must pass
- Only update tests if behavior intentionally changed
- No new features during refactoring

### 3. Focus and Scope
- ONLY analyze code in git diff
- Don't expand scope without asking
- Don't refactor unrelated code
- Stay focused on actual changes made

## Workflow

```
┌─────────────────────────────────────┐
│  1. Detect Git Changes              │
│     - git status                    │
│     - git diff / git diff --cached  │
│     - Identify changed files        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Analyze Changed Code            │
│     - Read changed files            │
│     - Focus on modified sections    │
│     - Check best practices          │
│     - Compare with project patterns │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Report Findings                 │
│     - Categorize: Critical/Mod/Minor│
│     - Provide specific fixes        │
│     - Create GIT_CHANGES_REVIEW.md  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Ask User                        │
│     - Apply all fixes?              │
│     - Apply critical only?          │
│     - Manual review?                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Refactor (if approved)          │
│     - One change at a time          │
│     - Test after each change        │
│     - Fix failures immediately      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Final Verification              │
│     - Run full test suite           │
│     - Review git diff               │
│     - Provide summary               │
└─────────────────────────────────────┘
```

## Common Use Cases

### Pre-Commit Review
```
Scenario: You've made changes and want to ensure quality before committing

Command: "Review my changes before I commit"

Result:
- Analyzes unstaged + staged changes
- Identifies issues in your modifications
- Fixes issues (if you approve)
- Leaves you with clean, tested changes
- Suggests commit message
```

### Security Review
```
Scenario: You made security-sensitive changes (auth, database, API)

Command: "Review my changes for security issues"

Result:
- Focuses on security aspects
- Checks for OWASP Top 10 vulnerabilities
- Validates input handling
- Ensures no secrets exposed
- Recommends security tests
```

### Feature Branch Review
```
Scenario: Large feature branch with many files changed

Command: "Review all changes in my feature branch"

Result:
- Assesses scope (offers phased review if large)
- Reviews by category or file
- Comprehensive quality check
- Fixes issues incrementally
- Prepares branch for PR/merge
```

### Post-Commit Cleanup
```
Scenario: Already committed but want to improve before pushing

Command: "Review my last 3 commits"

Result:
- Analyzes committed changes
- Suggests improvements
- Can amend or create new fix commit
- Ensures pushed code is high quality
```

## Issue Categories

### 🔴 Critical Issues (Must Fix)
- Security vulnerabilities (SQL injection, XSS, etc.)
- Data loss risks
- Breaking changes to public API
- Performance regressions (>10x slower)
- Incorrect error handling

**Action:** Fix immediately, don't commit without fixing

### 🟡 Moderate Issues (Should Fix)
- Missing type hints (inconsistent with codebase)
- Code duplication
- Missing error handling
- Performance inefficiencies (2-10x slower)
- Inconsistent patterns

**Action:** Fix before merging to main, acceptable to commit to feature branch

### 🟢 Minor Issues (Nice to Have)
- Naming improvements
- Missing docstrings
- Style inconsistencies
- Minor optimizations
- Refactoring opportunities

**Action:** Optional, can defer to later

## File Outputs

### GIT_CHANGES_REVIEW.md
Detailed analysis report with:
- Summary of changed files
- List of all issues found (categorized)
- Specific code examples and fixes
- Testing recommendations
- Next steps

## Integration with Other Skills

### Recommended Workflows

**Pre-Commit Quality:**
```
1. Make changes
2. review-git-changes → Fix issues
3. Commit with clean code
```

**Full Quality Audit:**
```
1. code-analysis → Understand full file quality
2. review-git-changes → Focus on your recent changes
3. refactor-code → Apply systematic improvements
```

**Production Preparation:**
```
1. review-git-changes → Clean up feature changes
2. best-practice → Make production-ready
3. Commit and deploy
```

**Security Critical:**
```
1. review-git-changes (security focus)
2. Manual security review
3. review-git-changes (verify fixes)
4. Commit
```

## Configuration

This skill uses:
- **allowed-tools:** Read, Edit, Bash(git/pytest/python), Grep, Glob, WebSearch
- **model:** Inherits from parent (Sonnet 4.5)

## Examples

See [EXAMPLES.md](EXAMPLES.md) for detailed usage examples:
1. Simple pre-commit review
2. Security issue detection
3. Large feature branch review
4. Post-commit refactoring
5. Quick syntax check

## Tips for Best Results

1. **Commit frequently** - Smaller changes are easier to review
2. **Run before pushing** - Catch issues before they reach remote
3. **Be specific** - Tell the skill what you're concerned about (security, performance, etc.)
4. **Trust the process** - Let the skill run tests after each change
5. **Review the diff** - Final git diff should make sense and be clean

## Limitations

- Only analyzes git-tracked files (won't see brand new untracked files until `git add`)
- Requires tests to verify refactoring (manual testing if no automated tests)
- Best for incremental changes (large rewrites might need full analysis)
- Focused on quality of changes, not architectural review

## FAQ

**Q: What's the difference between this and code-analysis?**

A: `code-analysis` looks at entire files. `review-git-changes` only looks at what you modified. Much faster and more focused.

**Q: Will this rewrite my code?**

A: Only if you approve. It always asks before making changes.

**Q: What if I only want to review, not refactor?**

A: Choose "Manual Review" when asked. You'll get a detailed report to use as a guide.

**Q: Can I review already-committed code?**

A: Yes! Use `git diff HEAD~N..HEAD` to review last N commits.

**Q: What if there are no tests?**

A: The skill will note this and recommend adding tests. It will verify refactoring manually where possible.

**Q: How long does a review take?**

A: Depends on changes:
- 1-2 files: ~2-5 minutes
- 5-10 files: ~10-15 minutes
- Large feature branch: ~20-30 minutes (or phased review)

## Contributing

To improve this skill:
1. Update SKILL.md with new patterns/best practices
2. Add examples to EXAMPLES.md
3. Test with real codebases
4. Share feedback on what works/doesn't work

## Version History

- **v1.0** - Initial release
  - Git change detection
  - Focused analysis
  - Incremental refactoring
  - Safety-first methodology
