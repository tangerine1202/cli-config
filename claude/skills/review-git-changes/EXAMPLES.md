# Review Git Changes - Usage Examples

This document provides concrete examples of the `review-git-changes` skill in action.

---

## Example 1: Simple Pre-Commit Review

### Scenario
Developer modified one file and wants to review before committing.

### User Command
```
"Review my changes before I commit"
```

### Skill Execution

**Step 1: Detect Changes**
```bash
$ git status
On branch feature/add-validation
Changes not staged for commit:
  modified:   src/user_service.py

$ git diff src/user_service.py
```

**Step 2: Analyze Changes**

```diff
+ def validate_email(email: str) -> bool:
+     if not email:
+         return False
+     if '@' not in email:
+         return False
+     return True
+
  def create_user(data):
-     email = data['email']
+     email = data.get('email')
+     if not validate_email(email):
+         raise ValueError('Invalid email')
      user = User(email=email)
      db.save(user)
```

**Issues Found:**
- 🟡 Moderate: Missing type hints on `create_user` (inconsistent with new function)
- 🟢 Minor: `validate_email` could use regex for better validation
- ✅ Good: Extracted validation logic
- ✅ Good: Safe dict access with `get()`

**Step 3: Report**

```markdown
# Git Changes Review Report

**Changed Files:** 1 file (src/user_service.py)
**Lines Added:** +8
**Lines Removed:** -1
**Quality Assessment:** Good (minor improvements suggested)

## Issues Found

### 🟡 Moderate Issues
1. **Inconsistent Type Hints** - `src/user_service.py:10`
   - **Problem:** New function has type hints, but modified function doesn't
   - **Impact:** Inconsistent code style, reduced type safety
   - **Fix:** Add type hints to `create_user`

### 🟢 Minor Issues
1. **Basic Email Validation** - `src/user_service.py:2`
   - **Problem:** Simple check, doesn't validate email format
   - **Impact:** Could accept invalid emails like "test@"
   - **Fix:** Use regex pattern or email validation library

## Positive Changes
- ✅ Extracted validation into separate function (good separation of concerns)
- ✅ Changed from `data['email']` to `data.get('email')` (safer)
- ✅ Added proper error handling with ValueError

## Next Steps
Apply moderate issue fix? The change is minimal and improves consistency.
```

**Step 4: User Chooses "Apply Moderate Issues"**

**Step 5: Refactor**
```python
def create_user(data: dict) -> User:
    email = data.get('email')
    if not validate_email(email):
        raise ValueError('Invalid email')
    user = User(email=email)
    db.save(user)
    return user
```

**Step 6: Verify**
```bash
$ pytest tests/test_user_service.py
=================== 5 passed in 0.23s ===================
```

**Step 7: Final Report**
```markdown
## Review Complete ✅

**Fixed:** 1 moderate issue (added type hints to create_user)
**Tests:** All passing (5/5)
**Status:** Ready to commit

Suggested commit message:
```
Add email validation to user creation

- Extract validation logic into validate_email()
- Add proper error handling for invalid emails
- Add type hints for consistency

🤖 Reviewed with Claude Code review-git-changes skill
```
```

---

## Example 2: Security Issue in Changes

### Scenario
Developer added a database query and wants security review.

### User Command
```
"Review my database changes for security issues"
```

### Skill Execution

**Step 1: Git Diff**
```diff
  def get_user_posts(user_id):
-     posts = db.query("SELECT * FROM posts WHERE user_id = ?", (user_id,))
+     # Added search functionality
+     search_term = request.args.get('search', '')
+     query = f"SELECT * FROM posts WHERE user_id = {user_id}"
+     if search_term:
+         query += f" AND title LIKE '%{search_term}%'"
+     posts = db.query(query)
      return posts
```

**Step 2: Analysis (Security Focused)**

**Issues Found:**
- 🔴 **CRITICAL**: SQL Injection vulnerability
- 🔴 **CRITICAL**: Removed parameterized query

**Step 3: Report**

```markdown
# Git Changes Review Report

**Changed Files:** 1 file (src/posts.py)
**Lines Added:** +5
**Lines Removed:** -1
**Quality Assessment:** ⚠️ CRITICAL SECURITY ISSUES

---

## 🚨 Critical Security Issues (MUST FIX IMMEDIATELY)

### 1. SQL Injection Vulnerability - `src/posts.py:15`

**Problem:**
```python
# DANGEROUS - User input directly in SQL
query += f" AND title LIKE '%{search_term}%'"
```

User can inject SQL by providing:
```
search_term = "'; DROP TABLE posts; --"
```

Resulting query:
```sql
SELECT * FROM posts WHERE user_id = 123 AND title LIKE '%'; DROP TABLE posts; --%'
```

**Impact:**
- Attacker can read entire database
- Attacker can delete all data
- Attacker can modify data
- **OWASP Top 10 #1 vulnerability**

**Fix:**
```python
# Safe - Parameterized query
query = "SELECT * FROM posts WHERE user_id = ? AND title LIKE ?"
params = [user_id, f"%{search_term}%"]
posts = db.query(query, params)
```

### 2. Removed Parameterization - `src/posts.py:14`

**Problem:** Removed safe parameterized query `(user_id,)`

**Original (SAFE):**
```python
db.query("SELECT * FROM posts WHERE user_id = ?", (user_id,))
```

**New (UNSAFE):**
```python
query = f"SELECT * FROM posts WHERE user_id = {user_id}"
```

**Impact:** Even without search, the user_id is now injectable

**Fix:** Use parameterized queries for ALL variables

---

## Recommended Fix (Complete)

```python
def get_user_posts(user_id):
    # Use parameterized query to prevent SQL injection
    query = "SELECT * FROM posts WHERE user_id = ?"
    params = [user_id]

    # Add search filter if provided
    search_term = request.args.get('search', '')
    if search_term:
        query += " AND title LIKE ?"
        params.append(f"%{search_term}%")

    posts = db.query(query, params)
    return posts
```

---

## Testing Recommendations

Add security tests:
```python
def test_sql_injection_prevention():
    """Ensure malicious input doesn't execute SQL"""
    malicious_input = "'; DROP TABLE posts; --"

    # Should not raise exception or delete data
    result = get_user_posts(user_id=1, search=malicious_input)

    # Verify posts table still exists
    assert db.table_exists('posts')

    # Verify no posts match (literal string match, not SQL execution)
    assert len(result) == 0
```

---

## IMMEDIATE ACTION REQUIRED

**DO NOT COMMIT THIS CODE**

These are critical security vulnerabilities. Fix immediately before proceeding.
```

**Step 4: User Chooses "Apply Critical Fixes"**

**Step 5: Refactor with Security Fix**
```python
def get_user_posts(user_id):
    """Get posts for a user, optionally filtered by search term.

    Args:
        user_id: The user ID to filter posts

    Returns:
        List of Post objects

    Security:
        Uses parameterized queries to prevent SQL injection
    """
    query = "SELECT * FROM posts WHERE user_id = ?"
    params = [user_id]

    search_term = request.args.get('search', '')
    if search_term:
        query += " AND title LIKE ?"
        params.append(f"%{search_term}%")

    posts = db.query(query, params)
    return posts
```

**Step 6: Verify**
```bash
$ pytest tests/test_posts.py -v
test_get_user_posts ... PASSED
test_get_user_posts_with_search ... PASSED
test_sql_injection_prevention ... PASSED

$ safety check  # Check for known vulnerabilities
All good!
```

**Step 7: Final Report**
```markdown
## Security Review Complete ✅

**Critical Issues Fixed:** 2
1. ✅ SQL injection in search filter
2. ✅ Removed unsafe string interpolation

**Security Tests Added:**
- ✅ SQL injection prevention test

**Status:** Safe to commit

**Commit Message:**
```
Add search functionality to user posts (SECURE)

- Add search filter for post titles
- Use parameterized queries to prevent SQL injection
- Add security test for injection prevention
- Add docstring with security notes

Security: Fixed SQL injection vulnerability before commit
🤖 Reviewed with Claude Code review-git-changes skill
```
```

---

## Example 3: Large Feature Branch Review

### Scenario
Developer has a feature branch with 15 files changed and wants comprehensive review.

### User Command
```
"Review all my changes in the feature/new-auth branch"
```

### Skill Execution

**Step 1: Assess Scope**
```bash
$ git diff main..feature/new-auth --stat
src/auth/login.py           | 45 +++++++++++++++++++++++-
src/auth/session.py         | 89 ++++++++++++++++++++++++++++++++++++++++++
src/auth/middleware.py      | 34 ++++++++++++++++
src/models/user.py          | 12 +++---
src/api/routes/auth.py      | 67 ++++++++++++++++++++++++++++++++
tests/test_auth.py          | 123 ++++++++++++++++++++++++++++++++++++++++++++++++++++
... (9 more files)

15 files changed, 487 insertions(+), 23 deletions(-)
```

**Response to User:**
```
I see 15 files changed with ~487 additions across your feature branch.
This is a large review. I can:

1. Review all files (comprehensive, will take ~20-30 min)
2. Review by category (auth files, tests, models separately)
3. Review specific files you're concerned about
4. Review commit-by-commit (5 commits in branch)

Which approach would you prefer?
```

**User Chooses:** "Review by category, start with auth files"

**Step 2: Focused Analysis (Auth Files)**

```markdown
# Git Changes Review Report - Auth Module

**Files Reviewed:** 3 files
- src/auth/login.py
- src/auth/session.py
- src/auth/middleware.py

**Lines:** +168 added, -5 removed

---

## Issues Found

### 🔴 Critical Issues

1. **Hardcoded Secret Key** - `src/auth/session.py:12`
   ```python
   SECRET_KEY = "my-secret-key-12345"  # ❌ NEVER hardcode
   ```
   **Fix:** Use environment variable
   ```python
   SECRET_KEY = os.environ.get('SESSION_SECRET_KEY')
   if not SECRET_KEY:
       raise ValueError("SESSION_SECRET_KEY not set")
   ```

2. **Missing Session Timeout** - `src/auth/session.py:45`
   ```python
   def create_session(user_id):
       token = generate_token(user_id)
       sessions[token] = user_id  # No expiration! ❌
       return token
   ```
   **Fix:** Add timeout
   ```python
   sessions[token] = {
       'user_id': user_id,
       'expires': datetime.now() + timedelta(hours=24)
   }
   ```

### 🟡 Moderate Issues

1. **Weak Password Check** - `src/auth/login.py:23`
   - Current: Only checks length > 8
   - Recommended: Check complexity, common passwords

2. **No Rate Limiting** - `src/auth/login.py:15`
   - Vulnerable to brute force attacks
   - Add rate limiting (e.g., max 5 attempts per minute)

### 🟢 Minor Issues

1. **Missing Type Hints** - Various files
2. **Inconsistent Error Messages** - Mix of "Invalid credentials" and "Login failed"

---

## Positive Changes
- ✅ Good separation of concerns (login, session, middleware)
- ✅ Proper password hashing with bcrypt
- ✅ Comprehensive test coverage (123 new tests)
- ✅ Good docstrings on public functions

---

## Next Steps

Fix critical issues immediately (hardcoded secret, no session timeout).
These are security vulnerabilities.

After fixing critical issues, I can review:
- Model changes (user.py)
- API routes
- Remaining files
```

**Step 3: User Approves Critical Fixes**

**Step 4: Apply Fixes**
```python
# src/auth/session.py
import os
from datetime import datetime, timedelta

SECRET_KEY = os.environ.get('SESSION_SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SESSION_SECRET_KEY environment variable must be set")

def create_session(user_id: int) -> str:
    """Create a new session for user.

    Session expires after 24 hours.
    """
    token = generate_token(user_id)
    sessions[token] = {
        'user_id': user_id,
        'created': datetime.now(),
        'expires': datetime.now() + timedelta(hours=24)
    }
    return token

def is_session_valid(token: str) -> bool:
    """Check if session exists and hasn't expired."""
    session = sessions.get(token)
    if not session:
        return False
    if datetime.now() > session['expires']:
        del sessions[token]  # Clean up expired session
        return False
    return True
```

**Step 5: Verify**
```bash
$ pytest tests/test_auth.py -v
test_create_session ... PASSED
test_session_expiration ... PASSED
test_expired_session_cleanup ... PASSED
test_secret_key_required ... PASSED
... (119 more passing)

$ ruff check src/auth/
All checks passed!
```

**Step 6: Continue with Other Files**
```
✅ Auth files reviewed and fixed (2 critical issues)

Ready to review next category:
- Model changes (user.py)
- API routes (routes/auth.py)
- Remaining 9 files

Continue?
```

---

## Example 4: Refactoring Existing Commit

### Scenario
Developer already committed code but wants to improve it before pushing.

### User Command
```
"Review my last commit and make sure it's clean"
```

### Skill Execution

**Step 1: Check Last Commit**
```bash
$ git log -1 --oneline
a3f8c92 Add data processing pipeline

$ git show a3f8c92 --stat
src/pipeline/processor.py | 89 ++++++++++++++++++++++++++++++++++++++
tests/test_pipeline.py     | 34 ++++++++++++++
2 files changed, 123 insertions(+)
```

**Step 2: Review Commit Diff**
```python
# src/pipeline/processor.py (committed code)
def process_data(data):
    results = []
    for item in data:
        result = {}
        result['id'] = item['id']
        result['value'] = item['value'] * 2
        result['status'] = 'processed'
        results.append(result)
    return results

def process_batch(items):
    processed = []
    for i in range(len(items)):
        item = items[i]
        p = process_data([item])
        processed.extend(p)
    return processed
```

**Step 3: Analysis**

```markdown
# Git Changes Review Report - Commit a3f8c92

**Commit:** a3f8c92 "Add data processing pipeline"
**Files:** 2 files (src/pipeline/processor.py, tests/test_pipeline.py)
**Lines:** +123

---

## Issues Found in Commit

### 🟡 Moderate Issues

1. **Inefficient Loop in process_batch** - `processor.py:11`
   ```python
   # Current: O(N²) - calling process_data for each item
   for i in range(len(items)):
       item = items[i]
       p = process_data([item])  # ❌ Inefficient
       processed.extend(p)
   ```
   **Fix:** Call process_data once
   ```python
   # O(N) - process all items at once
   processed = process_data(items)
   ```

2. **Dict Construction Pattern** - `processor.py:4-7`
   ```python
   # Verbose
   result = {}
   result['id'] = item['id']
   result['value'] = item['value'] * 2
   result['status'] = 'processed'
   ```
   **Fix:** Use dict literal
   ```python
   result = {
       'id': item['id'],
       'value': item['value'] * 2,
       'status': 'processed'
   }
   ```

3. **Unnecessary range(len())** - `processor.py:11`
   ```python
   for i in range(len(items)):  # ❌ Unpythonic
       item = items[i]
   ```
   **Fix:** Direct iteration
   ```python
   for item in items:  # ✅ Pythonic
   ```

### 🟢 Minor Issues

1. **Missing Type Hints** - All functions
2. **Missing Docstrings** - All functions

---

## Refactored Version

```python
def process_data(data: list[dict]) -> list[dict]:
    """Process data items by doubling their values.

    Args:
        data: List of items with 'id' and 'value' keys

    Returns:
        List of processed items with doubled values
    """
    return [
        {
            'id': item['id'],
            'value': item['value'] * 2,
            'status': 'processed'
        }
        for item in data
    ]

def process_batch(items: list[dict]) -> list[dict]:
    """Process a batch of items.

    Args:
        items: List of items to process

    Returns:
        List of processed items
    """
    return process_data(items)
```

---

## Performance Impact

**Before:** O(N²) with N function calls
**After:** O(N) with 1 function call

For 1000 items:
- Before: ~1000 iterations + 1000 function calls
- After: ~1000 iterations + 1 function call

**Estimated speedup:** 10-50x for large batches

---

## Next Steps

**Option 1: Amend Commit** (rewrite history)
```bash
git add src/pipeline/processor.py
git commit --amend --no-edit
```

**Option 2: New Commit** (preserve history)
```bash
git add src/pipeline/processor.py
git commit -m "Refactor: Optimize data processing pipeline"
```

**Recommendation:**
Since you haven't pushed yet, **Option 1 (amend)** keeps history clean.
If already pushed to shared branch, use **Option 2 (new commit)**.
```

**Step 4: User Chooses "Amend Commit"**

**Step 5: Apply Refactoring**
[Apply the refactored code using Edit tool]

**Step 6: Verify**
```bash
$ pytest tests/test_pipeline.py
=================== 5 passed in 0.12s ===================

$ python -m pytest tests/test_pipeline.py --benchmark
process_batch (1000 items): 0.05s (was 2.3s) ✅ 46x faster
```

**Step 7: Amend Commit**
```bash
$ git add src/pipeline/processor.py
$ git commit --amend --no-edit
[feature/pipeline a3f8c92] Add data processing pipeline
 2 files changed, 123 insertions(+), 45 deletions(-)
```

**Final Report:**
```markdown
## Commit Refactored Successfully ✅

**Original Commit:** a3f8c92 "Add data processing pipeline"
**Action:** Amended with improvements

**Improvements Applied:**
- ✅ Fixed O(N²) performance issue → O(N)
- ✅ Added type hints
- ✅ Added docstrings
- ✅ More Pythonic code patterns

**Performance:**
- 46x faster for large batches (tested with 1000 items)

**Tests:** All passing (5/5)

**Status:** Ready to push
```

---

## Example 5: Quick Syntax Check

### Scenario
Developer made quick changes and just wants to ensure no obvious issues.

### User Command
```
"Quick check on my changes"
```

### Skill Execution

**Step 1: Quick Diff**
```bash
$ git diff
diff --git a/src/utils.py b/src/utils.py
-def calculate_total(items):
+def calculate_total(items: list[dict]) -> float:
+    """Calculate total price of items."""
     return sum(item['price'] for item in items)
```

**Step 2: Quick Analysis**

```markdown
# Quick Review ✅

**Changed:** src/utils.py (3 lines)

**Issues:** None found

**Improvements Made:**
- ✅ Added type hints (list[dict] -> float)
- ✅ Added docstring
- ✅ Maintained existing logic

**Status:** Good to commit!

No issues detected. The changes improve code quality without changing behavior.
```

---

## Summary

These examples demonstrate:

1. **Pre-commit review** - Catching issues before they enter history
2. **Security-focused review** - Preventing critical vulnerabilities
3. **Large feature review** - Managing scope for big changes
4. **Post-commit cleanup** - Improving already-committed code
5. **Quick checks** - Fast validation for small changes

The skill adapts its depth and focus based on:
- Size of changes (1 file vs 15 files)
- Type of code (security-critical vs utility functions)
- User's goals (quick check vs thorough review)
- Severity of issues (critical security vs minor style)
