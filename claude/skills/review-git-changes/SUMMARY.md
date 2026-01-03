# Review Git Changes Skill - Summary

## What Was Created

A new Claude Code skill focused on analyzing and refactoring **only the code changes in your git working tree**. This skill ensures your modifications follow best practices before you commit them to version control.

### Files Created

```
.claude/skills/review-git-changes/
├── SKILL.md (593 lines)      - Core skill definition and methodology
├── EXAMPLES.md (801 lines)    - 5 detailed usage scenarios
├── README.md (309 lines)      - Quick start guide and documentation
└── SUMMARY.md (this file)     - Overview and comparison
```

---

## How It Differs from Existing Skills

### Quick Comparison Table

| Skill | Scope | When to Use | Speed | Output |
|-------|-------|-------------|-------|--------|
| **review-git-changes** | Git diff only | Before commit, reviewing changes | Fast (2-15 min) | GIT_CHANGES_REVIEW.md |
| **code-analysis** | Entire files | Full quality check | Medium (15-30 min) | CODE_QUALITY_REPORT.md |
| **refactor-code** | Implementation plan | Systematic refactoring | Medium (20-40 min) | Refactored files |
| **critic** | Full audit + refactor | Complete quality workflow | Slow (30-60 min) | Plan + refactored files |

### Detailed Differences

#### review-git-changes (NEW)
**Focus:** Only the code you changed
**Strength:** Fast, focused, pre-commit validation
**Use when:**
- "Review my changes before I commit"
- "Check what I changed is good"
- "Ensure my git diff is clean"

**Example:**
```bash
# You modified 3 files
git status
# modified: src/auth.py
# modified: src/db.py
# modified: tests/test_auth.py

# Skill analyzes ONLY those 3 files AND ONLY the changed sections
# Much faster than analyzing entire codebase
```

#### code-analysis (EXISTING)
**Focus:** Entire files or modules
**Strength:** Comprehensive quality assessment
**Use when:**
- "Analyze this file for quality"
- "Check code quality of module"
- "Quick quality assessment"

**Example:**
```bash
# Analyzes entire factory_env.py (1265 lines)
# Even if you only changed 10 lines
```

#### refactor-code (EXISTING)
**Focus:** Applying improvements from a plan
**Strength:** Systematic, thorough refactoring
**Use when:**
- "Refactor based on implementation_plan.md"
- "Apply these improvements"
- "Make code more maintainable"

**Example:**
```bash
# Requires implementation_plan.md to exist
# Applies all recommended changes systematically
```

#### critic (EXISTING)
**Focus:** Analysis + Refactoring workflow
**Strength:** Complete end-to-end audit
**Use when:**
- "Critic my code"
- "Full quality audit"
- "Review and improve everything"

**Example:**
```bash
# Full workflow: analyze → plan → refactor
# Most comprehensive but takes longest
```

---

## Key Principles (From refactor-code)

The new skill follows the same safety-first methodology:

### 1. Safety First ✅
```
✓ Never skip tests
✓ Make changes incrementally (one at a time)
✓ Run verification after EACH change
✓ Keep working versions (easy rollback with git)
✓ If uncertain, ask user before major changes
```

### 2. Maintain Functionality ✅
```
✓ Refactoring should NOT change behavior
✓ All existing tests must pass
✓ Don't introduce new features while refactoring
✓ Don't remove functionality without approval
```

### 3. Focus and Scope ✅
```
✓ Refactor only what was analyzed or requested
✓ Don't expand scope without asking
✓ Don't "improve" code that's not part of task
✓ Don't add unnecessary abstractions
```

**Special addition for git-focused skill:**
```
✓ ONLY analyze code in git diff (not entire files)
✓ Don't refactor unrelated code
✓ Stay focused on actual changes made
✓ Respect user's original intent
```

---

## How to Use the New Skill

### Basic Usage

```bash
# 1. Make some code changes
vim src/myfile.py

# 2. Ask Claude to review
"Review my git changes"

# Or be more specific:
"Review my changes before commit"
"Check what I changed for security issues"
"Ensure my changes follow best practices"
```

### What Happens

```
Step 1: Detect Changes
- Runs git status, git diff
- Identifies changed files
- Asks for scope clarification if needed

Step 2: Analyze Changes
- Reads changed files
- Focuses on modified sections only
- Checks best practices
- Identifies issues: Critical/Moderate/Minor

Step 3: Report Findings
- Creates GIT_CHANGES_REVIEW.md
- Shows issues with code examples
- Suggests specific fixes

Step 4: Ask for Approval
- "Apply all fixes?"
- "Apply critical only?"
- "Manual review?"

Step 5: Refactor (if approved)
- One change at a time
- Test after each change
- Fix failures immediately

Step 6: Final Verification
- Run full test suite
- Review final git diff
- Provide summary
```

### Example Session

```
User: "Review my changes before I commit"

Claude:
I see you have changes in 2 files:
- src/auth.py: +45 lines, -12 lines
- tests/test_auth.py: +34 lines

Analyzing changes...

Found 1 critical issue:
🔴 SQL Injection vulnerability in auth.py:45
   - Using string interpolation instead of parameterized query
   - Fix: Use db.query("SELECT * FROM users WHERE id = ?", (user_id,))

Found 2 moderate issues:
🟡 Missing type hints on new function authenticate()
🟡 New exception not handled in caller

Would you like me to fix these issues? (All / Critical only / Manual)

User: "Fix all"

Claude:
Applying fixes...
✅ Fixed SQL injection (auth.py:45)
✅ Added type hints (auth.py:23)
✅ Added exception handling (auth.py:67)

Running tests...
✅ All tests passing (12/12)

Git diff review:
✅ Changes look clean
✅ Ready to commit

Suggested commit message:
---
Add user authentication with secure database queries

- Implement authenticate() function with type hints
- Use parameterized queries to prevent SQL injection
- Add proper exception handling

🤖 Reviewed with Claude Code review-git-changes skill
---
```

---

## Advanced Features

### 1. Security-Focused Review
```
"Review my changes for security issues"

→ Focuses specifically on:
  - SQL injection
  - XSS vulnerabilities
  - Command injection
  - Hardcoded secrets
  - Input validation
```

### 2. Large Feature Branch Review
```
"Review all changes in feature/new-auth branch"

→ Handles large scope:
  - Offers phased review (by file category)
  - Prioritizes critical issues
  - Can review commit-by-commit
```

### 3. Post-Commit Cleanup
```
"Review my last 3 commits"

→ Analyzes already-committed code:
  - Can amend last commit (if not pushed)
  - Can create new fix commit
  - Ensures pushed code is high quality
```

### 4. Quick Validation
```
"Quick check on my changes"

→ Fast validation:
  - Minimal analysis
  - Catches obvious issues
  - Quick pass/fail assessment
```

---

## Integration with Existing Workflows

### Workflow 1: Pre-Commit Quality
```
1. Make changes to code
2. /review-git-changes → Fix issues in changes
3. git commit -m "Clean commit message"
```

### Workflow 2: Full Quality Audit
```
1. /code-analysis → Understand full file quality
2. /review-git-changes → Focus on recent changes
3. /refactor-code → Apply systematic improvements
```

### Workflow 3: Feature Development
```
1. Develop feature (multiple commits)
2. /review-git-changes → Review entire branch
3. Fix issues
4. Create pull request with clean code
```

### Workflow 4: Security Critical
```
1. Make security-sensitive changes
2. /review-git-changes (security focus)
3. Manual security review
4. /review-git-changes (verify fixes)
5. Commit and deploy
```

---

## Why This Skill Was Created

### Problem It Solves

**Before:**
- Code analysis looks at entire files (slow, lots of noise)
- Hard to focus on just what you changed
- Easy to commit bad code without review
- Post-commit fixes pollute git history

**After:**
- Fast, focused analysis of only your changes
- Catches issues before they enter git history
- Clean commits with confidence
- Better code quality over time

### Design Philosophy

1. **Git-Native**: Works with git workflow naturally
2. **Focused**: Only what you changed, not the entire codebase
3. **Safe**: Never breaks working code
4. **Fast**: Quick enough to use before every commit
5. **Actionable**: Specific fixes, not vague advice

---

## Technical Details

### Skill Definition (SKILL.md)

**Metadata:**
```yaml
name: review-git-changes
description: Analyzes and refactors code changes in git working tree...
allowed-tools: Read, Edit, Bash(git:*), Bash(pytest:*), Grep, Glob, WebSearch
```

**Structure:**
- When to Use (triggers)
- Workflow Overview (6 phases)
- Step-by-Step Process (detailed methodology)
- Important Guidelines (safety, functionality, scope)
- Common Scenarios (5 examples)
- Integration with Other Skills

### Example Scenarios (EXAMPLES.md)

1. **Simple Pre-Commit Review** - Basic usage
2. **Security Issue Detection** - SQL injection prevention
3. **Large Feature Branch** - Phased review approach
4. **Post-Commit Refactoring** - Amending commits
5. **Quick Syntax Check** - Fast validation

### Documentation (README.md)

- Quick start guide
- Comparison table with other skills
- Workflow diagram
- Common use cases
- FAQ section
- Tips for best results

---

## Activation

The skill is automatically activated when you use phrases like:
- "Review my git changes"
- "Check what I changed"
- "Review before commit"
- "Ensure my changes are clean"
- "Refactor my git diff"

You can also invoke it explicitly:
```
/review-git-changes
```

---

## Testing the Skill

### Quick Test

```bash
# 1. Make a simple change
echo "# TODO: Fix this" >> src/test.py

# 2. Ask Claude
"Review my git changes"

# 3. Should detect:
# - TODO comment (minor issue)
# - Missing implementation (moderate issue)

# 4. Apply fix or skip
"Manual review"

# 5. Verify report created
ls GIT_CHANGES_REVIEW.md
```

### Comprehensive Test

```bash
# 1. Create feature branch with multiple changes
git checkout -b test/review-skill

# 2. Make changes with intentional issues:
# - SQL injection vulnerability (critical)
# - Missing type hints (moderate)
# - Magic numbers (minor)

# 3. Review
"Review all my changes in this branch"

# 4. Verify skill:
# - Detects all issues
# - Categorizes correctly
# - Provides specific fixes
# - Asks for approval
# - Applies fixes incrementally
# - Runs tests after each change
# - Provides final summary
```

---

## Next Steps

### For Immediate Use

1. **Try it out:**
   ```
   Make a small change
   → "Review my git changes"
   → See it in action
   ```

2. **Read examples:**
   ```
   Open EXAMPLES.md
   → See 5 detailed scenarios
   → Understand what to expect
   ```

3. **Integrate into workflow:**
   ```
   Before every commit:
   → "Review my changes"
   → Fix issues
   → Commit with confidence
   ```

### For Customization

1. **Adjust severity thresholds:**
   Edit SKILL.md to change what counts as Critical/Moderate/Minor

2. **Add project-specific checks:**
   Add custom patterns to look for in your codebase

3. **Customize output format:**
   Modify report structure in SKILL.md

4. **Add more examples:**
   Document your usage patterns in EXAMPLES.md

---

## Summary

**What was created:**
A focused git-aware code review skill that analyzes only your changes

**Key principles followed:**
Safety first, maintain functionality, focus & scope (from refactor-code)

**Main differences:**
Analyzes git diff (not entire files), much faster, pre-commit focused

**How to use:**
"Review my git changes" before committing

**When to use:**
Before commits, for feature branches, for security reviews, quick checks

**Integration:**
Works with existing skills (code-analysis, refactor-code, critic)

**Files:**
- SKILL.md (core methodology)
- EXAMPLES.md (5 usage scenarios)
- README.md (quick start guide)

**Status:**
✅ Ready to use immediately

---

## Questions?

**How is this different from code-analysis?**
→ Only looks at git changes, not entire files. Much faster.

**Will it change my code without asking?**
→ No. Always asks for approval before making changes.

**Can I review already-committed code?**
→ Yes. Use git diff HEAD~N..HEAD to review last N commits.

**What if there are no tests?**
→ Skill will note this and verify changes manually where possible.

**How long does a review take?**
→ 2-5 min for small changes, 10-15 min for medium, 20-30 min for large branches.

---

## Feedback Welcome

This is v1.0 of the skill. Please provide feedback on:
- What works well
- What could be improved
- Additional features needed
- Examples that would be helpful

The skill will evolve based on real-world usage!
