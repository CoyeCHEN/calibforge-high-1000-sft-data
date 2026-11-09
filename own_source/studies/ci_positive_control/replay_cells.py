"""Run inside a bounded, networkless Docker container, one IPython state/QA."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()
records = []
for line in Path("/traces.jsonl").open():
    trace = json.loads(line)
    shell.run_line_magic("reset", "-f")
    for i, message in enumerate(trace["source"]["messages"]):
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if function["name"] != "add_and_execute_jupyter_code_cell":
                continue
            code = function["arguments"]["code"]
            before = sorted(k for k in shell.user_ns if not k.startswith("_"))
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = shell.run_cell(code, store_history=False)
            entry = dict(trace_id=trace["source"]["id"], message_index=i, code=code,
                         prior_namespace=before, stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                         success=result.success, value=repr(result.result),
                         source_observation=trace["source"]["messages"][i+1]["content"])
            records.append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if not result.success:
                raise RuntimeError("Source cell failed during faithful bounded replay")
Path("/workspace/replay.json").write_text(json.dumps(records,indent=2)+"\n")
