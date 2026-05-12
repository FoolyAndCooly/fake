#!/usr/bin/env python
"""Add temporary debug logs to kernel modules to verify which path is taken.

This script patches fake/kernels/modules.py to add print statements at the
entry of _kernel_forward and _fallback_forward methods.

Usage:
    python scripts/add_kernel_debug_logs.py apply    # Add debug logs
    python scripts/add_kernel_debug_logs.py remove   # Remove debug logs
"""
from __future__ import annotations

import sys
from pathlib import Path


MODULES_PATH = Path("fake/kernels/modules.py")

KERNEL_FORWARD_MARKER = "def _kernel_forward(self"
FALLBACK_FORWARD_MARKER = "def _fallback_forward(self"

DEBUG_LOG_KERNEL = '        print(f"[DEBUG] {self.__class__.__name__}._kernel_forward called")'
DEBUG_LOG_FALLBACK = '        print(f"[DEBUG] {self.__class__.__name__}._fallback_forward called")'


def add_debug_logs() -> None:
    """Add debug print statements to _kernel_forward and _fallback_forward."""
    if not MODULES_PATH.exists():
        print(f"Error: {MODULES_PATH} not found")
        sys.exit(1)

    content = MODULES_PATH.read_text()
    lines = content.splitlines()
    modified_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        modified_lines.append(line)

        # After "def _kernel_forward(self, ...):" add debug log
        if KERNEL_FORWARD_MARKER in line and DEBUG_LOG_KERNEL not in content:
            # Find the first line of the function body (skip docstring if present)
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
                modified_lines.append(lines[i])
                i += 1
                # Skip to end of docstring
                if '"""' in lines[i-1] or "'''" in lines[i-1]:
                    while i < len(lines) and not (lines[i].strip().endswith('"""') or lines[i].strip().endswith("'''")):
                        modified_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        modified_lines.append(lines[i])
                        i += 1
            # Insert debug log
            modified_lines.append(DEBUG_LOG_KERNEL)
            continue

        # After "def _fallback_forward(self, ...):" add debug log
        if FALLBACK_FORWARD_MARKER in line and DEBUG_LOG_FALLBACK not in content:
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
                modified_lines.append(lines[i])
                i += 1
                if '"""' in lines[i-1] or "'''" in lines[i-1]:
                    while i < len(lines) and not (lines[i].strip().endswith('"""') or lines[i].strip().endswith("'''")):
                        modified_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        modified_lines.append(lines[i])
                        i += 1
            modified_lines.append(DEBUG_LOG_FALLBACK)
            continue

        i += 1

    MODULES_PATH.write_text("\n".join(modified_lines) + "\n")
    print(f"✅ Debug logs added to {MODULES_PATH}")
    print("   Run your script and look for '[DEBUG] ClassName._kernel_forward called' in output")


def remove_debug_logs() -> None:
    """Remove debug print statements from modules.py."""
    if not MODULES_PATH.exists():
        print(f"Error: {MODULES_PATH} not found")
        sys.exit(1)

    content = MODULES_PATH.read_text()
    lines = content.splitlines()
    modified_lines = [line for line in lines if DEBUG_LOG_KERNEL not in line and DEBUG_LOG_FALLBACK not in line]

    MODULES_PATH.write_text("\n".join(modified_lines) + "\n")
    print(f"✅ Debug logs removed from {MODULES_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "remove"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "apply":
        add_debug_logs()
    else:
        remove_debug_logs()
