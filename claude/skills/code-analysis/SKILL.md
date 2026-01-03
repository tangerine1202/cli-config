---
name: code-analysis
description: Quick code quality analysis that identifies common issues, anti-patterns, and improvement opportunities. Generates a concise summary report with actionable priorities. Use when the user asks to "analyze this code", "check code quality", "review this", or "what's wrong with this code".
allowed-tools: Read, Glob, Grep, WebSearch
---

# Code Analysis Skill

Fast code quality analysis that identifies issues and provides actionable recommendations in a concise format.

## When to Use This Skill

Trigger when users ask to:
- Analyze code quality
- Check for common issues
- Review code for problems
- Identify anti-patterns
- Get a quick quality assessment

**Not for:** Deep architectural analysis or applying fixes (use best-practice agent for post-MVP improvements)

## Analysis Methodology

### 1. Understand the Code (5-10 min)

**Read and map the structure:**
- Use Read tool to examine target files
- Identify main components and their responsibilities
- Note the technology stack (language, frameworks, libraries)
- Understand data flow and dependencies

**Questions to answer:**
- What does this code do?
- What are the main components?
- What patterns are being used?
- What external dependencies exist?

### 2. Identify Common Issues (10-15 min)

Check for these categories systematically:

#### Anti-Patterns

**Code Structure:**
- **Hardcoded values**: Magic numbers, hardcoded paths, embedded secrets
- **Code duplication**: Repeated logic that should be extracted
- **God objects/functions**: Components doing too many things (>100 lines, >3 responsibilities)
- **Deep nesting**: Excessive indentation (>3-4 levels)
- **Long parameter lists**: Functions with >5 parameters
- **Primitive obsession**: Using primitives instead of domain types

**Design Principles (SOLID):**
- **Single Responsibility**: Does each component have one clear purpose?
- **Open/Closed**: Can you extend without modifying existing code?
- **Liskov Substitution**: Are abstractions properly designed?
- **Interface Segregation**: Are interfaces focused and minimal?
- **Dependency Inversion**: Depending on abstractions, not concrete implementations?

**Other Principles:**
- **DRY (Don't Repeat Yourself)**: Is logic duplicated across files?
- **Separation of Concerns**: Are different aspects properly isolated?
- **Law of Demeter**: Avoiding excessive chaining (a.b.c.d.e)?

#### Performance Issues

- **Algorithm complexity**: O(N²) where O(N) or O(log N) is possible
- **Unnecessary iterations**: Multiple loops over the same data
- **Inefficient data structures**: Lists for lookups instead of hash maps/sets
- **Redundant computations**: Calculating the same value multiple times
- **Memory issues**: Loading large datasets entirely into memory

#### Maintainability Issues

- **Poor naming**: Unclear variable/function/class names
- **Missing error handling**: No validation, no error recovery
- **Inconsistent patterns**: Different approaches for the same problem
- **Tight coupling**: Hard to test or reuse components independently
- **Missing documentation**: Public APIs without docstrings/comments

#### Security Issues

- **Injection vulnerabilities**: SQL injection, command injection, XSS
- **Input validation**: Missing or insufficient validation
- **Secret exposure**: Hardcoded credentials, API keys in code
- **Insecure defaults**: Weak configurations, disabled security features

### 3. Quick Best Practice Check (5 min)

**Internal consistency:**
- Use Grep to find how similar problems are solved elsewhere in the codebase
- Example: `Grep -pattern "error handling|try.*catch|raise|throw" -output_mode content`
- Check if the code follows established project patterns

**External validation (selective):**
- If major issues found, do ONE targeted web search
- Focus on official documentation for the specific framework/library
- Example: "Python exception handling best practices" or "[Framework] error handling"

**Don't spend too long here** - this is quick analysis, not deep research.

### 4. Generate Quality Score and Report

Create a concise summary with this structure:

```markdown
# Code Quality Report

**Files Analyzed:** [list]
**Quality Score:** X/100

## Score Breakdown
- Structure & Design: X/30
- Performance: X/20
- Maintainability: X/30
- Security: X/20

---

## Top Priority Issues

### 🔴 Critical (Must Fix)
1. **[Issue Name]** - [Location]
   - **Problem:** [What's wrong]
   - **Impact:** [Why it matters]
   - **Quick Fix:** [1-line suggestion]

### 🟡 Important (Should Fix)
[2-3 items, same format]

### 🟢 Minor (Nice to Have)
[2-3 items, same format]

---

## Detailed Findings

### Code Structure Issues
- [Specific finding with file:line reference]
- [Specific finding with file:line reference]

### Performance Issues
- [Specific finding with file:line reference]

### Maintainability Issues
- [Specific finding with file:line reference]

### Security Issues
- [Specific finding with file:line reference]

---

## Quick Wins (Fix in <30 min)
1. [Easy fix that has good impact]
2. [Easy fix that has good impact]
3. [Easy fix that has good impact]

## Positive Observations
- [Things done well]
- [Good patterns observed]

---

## Next Steps

**If this is MVP/prototype code:**
- Fix critical issues now
- Consider using the `best-practice` agent to make this production-ready

**If this is production code:**
- Address critical and important issues
- Plan refactoring for structural improvements

---

## References
- [Internal pattern examples: file:line]
- [External documentation if searched]
```

## Scoring Guidelines

### Structure & Design (30 points)
- **25-30**: Clean separation of concerns, follows SOLID, minimal duplication
- **15-24**: Some violations, but generally organized
- **5-14**: Significant structural issues, god objects, tight coupling
- **0-4**: Severe architectural problems

### Performance (20 points)
- **17-20**: Efficient algorithms and data structures
- **10-16**: Generally okay, minor inefficiencies
- **5-9**: Notable bottlenecks or complexity issues
- **0-4**: Critical performance problems

### Maintainability (30 points)
- **25-30**: Clear naming, good error handling, well-documented
- **15-24**: Mostly maintainable, some unclear areas
- **5-14**: Difficult to understand or modify
- **0-4**: Unmaintainable code

### Security (20 points)
- **17-20**: Secure by design, proper validation
- **10-16**: Basic security, minor gaps
- **5-9**: Missing validations, potential vulnerabilities
- **0-4**: Critical security issues

**Overall Score:**
- **80-100**: Production-ready quality
- **60-79**: Good quality, minor improvements needed
- **40-59**: Acceptable for MVP, needs improvement
- **20-39**: Significant issues, refactoring recommended
- **0-19**: Critical quality problems

## Important Guidelines

**Be Fast:**
- Target: 15-30 minutes total analysis time
- Focus on obvious issues, not edge cases
- Don't do deep research - save that for best-practice agent

**Be Specific:**
- Always include file:line references
- Show concrete examples of issues
- Quantify when possible ("5 duplicated functions" not "some duplication")

**Be Actionable:**
- Prioritize by impact and effort
- Suggest specific fixes, not vague advice
- Highlight "quick wins" clearly

**Be Honest:**
- Use the scoring system objectively
- Point out both good and bad
- Don't sugarcoat critical issues

**Stay Focused:**
- Analyze only what user requested
- Don't expand scope without asking
- Don't try to fix issues (that's refactor-code's job)

## Output Format

Present findings to the user in this order:
1. **Quality score** (give them the headline first)
2. **Top 3-5 priority issues** (what needs attention)
3. **Quick wins** (easy improvements)
4. **Next steps recommendation**

Save detailed report to `CODE_QUALITY_REPORT.md`

## Example Workflow

**User:** "Analyze factory_env.py for code quality"

**Skill Actions:**
1. Read `factory_env.py` (1265 lines)
2. Scan for issues:
   - Find: 2 god methods (>200 lines each)
   - Find: 5 hardcoded magic numbers
   - Find: Performance issue in loop (O(N²))
   - Find: Good error handling, good naming
3. Quick grep for project patterns
4. Generate report:
   - Score: 62/100
   - Critical: 1 (performance bottleneck)
   - Important: 3 (god methods, magic numbers)
   - Minor: 2 (minor duplications)
5. Present summary to user
6. Save to `CODE_QUALITY_REPORT.md`

**Total time:** ~20 minutes

---

See `EXAMPLES.md` for scenario-specific reference examples.
