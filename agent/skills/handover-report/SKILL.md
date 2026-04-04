---
name: handover-report
description: Generates architecture-first technical handover reports for projects and components. Prioritizes the "Mental Model" (conceptual lifecycle and orchestration) before diving into structural tracing. Use this whenever the user asks for a "handover", "transition", "document architecture", or "comprehensive documentation" for a system they didn't build.
---

# Handover Report Skill

This skill allows agents to generate high-quality, architecture-first technical handover reports. The goal is to provide an incoming developer with a clear **mental model** of the system, followed by actionable **structural tracing** and **operational guides**.

## Core Principles

### 0. Analyze First

**MANDATORY**: You MUST perform a thorough analysis of the relevant codebase (using skills if available) BEFORE you begin drafting any part of the handover report. Do not rely on high-level summaries or your own internal knowledge; identify the exact files, classes, and orchestration patterns present in the current workspace.

### 1. Mental Model First

Before listing files or functions, explain the **Conceptual Lifecycle** or **Architecture**. The reader should understand the "Big Picture" (how data flows, how the system starts, what the main loops are) within the first 2 pages. Use Mermaid.js diagrams for visual clarity.

### 2. Structural Tracing (The Bridge)

Every technical concept must be "bridged" to the code. Never just describe a feature; point to the file path and function that implements it.

- **Example**: "The authentication logic ([auth.py](file:///path/to/auth.py)) uses JWT tokens..."

### 3. Conceptual Bridging (The "Why")

Explain design decisions and constraints. Why was this library chosen? Why is this specific algorithm used? Documentation of intent is more valuable than documentation of syntax.

### 4. Operational Readiness

Include a "Quick Start" or "Operations" section with verified CLI commands. If after a thorough analysis of the codebase and existing documentation the correct CLI commands or setup procedures are not explicitly found, **STOP and ask the user for help**. Do not guess the command-line arguments or environment setup.

## Report Structure Template

### I. Executive Summary

High-level purpose of the project and current status.

### II. The Mental Model (Architecture)

- **High-Level Diagram**: Use Mermaid.js if applicable.
- **System Lifecycle**: How does it start? How does it end?
- **Core Orchestration**: Identify the "Brain" or "Manager" classes.

### III. Technical Deep-Dive (Bridged)

Break down the system into logical layers (e.g., Data, Logic, UI, Physics).
For each layer:

- **Concept**: High-level explanation.
- **Implementation**: List of files/functions with links.
- **Key Parameters**: Magic numbers, configs, or non-obvious settings.

### IV. Operations & CLI

- **Installation**: One-liners if possible.
- **Commands**: Run, Test, Evaluate, Build, Deploy.
- **Directory Structure**: Visual tree of the repository.

## Domain-Specific Guidance

If the project belongs to a specific domain, refer to the `references/` directory for domain-specific guidance.
