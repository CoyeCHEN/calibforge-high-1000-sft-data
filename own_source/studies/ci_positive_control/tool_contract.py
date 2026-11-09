"""Reuse the pinned Harness tool implementations; preserve native CI tools."""
import json
from pathlib import Path
import sys

WHEEL = "open_harness_ai-0.3.0+rsi.summary.20261030-py3-none-any.whl"


def native_clean(value):
    """Arrow unions pad absent optional members with null. Drop only those pads."""
    if isinstance(value, dict):
        return {key: native_clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [native_clean(item) for item in value]
    return value


def terminal_tools(root, cwd):
    wheel = str(Path(root) / "source" / WHEEL)
    if wheel not in sys.path:
        sys.path.insert(0, wheel)
    from foundation.jobs import BashJobs
    from rsi_modules.capabilities.tools import createCodingTools
    from rsi_modules.capabilities.tools.bash_job import createBashJobTool
    from rsi_modules.capabilities.tools.finish import createFinishTool
    jobs = BashJobs(Path(cwd) / ".background-jobs", maxConcurrent=2)
    tools = createCodingTools(str(cwd), {"bash": {"jobs": jobs, "timeout": 120}})
    tools += [createBashJobTool(jobs), createFinishTool("Check the requested result before completing.", jobs)]
    assert [t["name"] for t in tools] == ["read", "bash", "edit", "write", "bash_job", "finish"]
    return tools, jobs


def schemas(root, native):
    terminal = json.loads((Path(root) / "configs/terminal-tools.json").read_text())
    result = terminal + native_clean(native)
    names = [t["function"]["name"] for t in result]
    assert len(names) == len(set(names)) == 8
    assert names[-2:] == ["add_and_execute_jupyter_code_cell", "final_answer"]
    return result


def messages(source, trace_id):
    """Restore unambiguous sequential IDs absent from the published ChatML rows."""
    result = []
    pending = None
    for index, original in enumerate(source):
        row = native_clean(original)
        if row["role"] == "assistant" and row.get("tool_calls"):
            assert pending is None
            assert len(row["tool_calls"]) == 1
            call = row["tool_calls"][0]
            call.update(id=f"ci-{trace_id}-{index}", type="function")
            pending = call["id"]
        elif row["role"] == "tool":
            assert pending is not None
            row["tool_call_id"] = pending
            pending = None
        result.append(row)
    return result
