"""Freeze official CIBench end-to-end datasets and concrete model configs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main(root, edition):
    root = Path(root)
    libraries = ["lightgbm", "matplotlib", "nltk", "opencv", "pandas", "pytorch",
                 "scipy", "seaborn", "sklearn", "tensorflow"]
    scopes = [(kind, lib) for kind in ["cibench_template", "cibench_template_chinese"] for lib in libraries]
    scopes += [("cibench_generation", lib) for lib in ["matplotlib", "opencv", "pandas", "pytorch", "scipy", "seaborn"]]
    entries = []
    for kind, lib in scopes:
        path = root / "data/cibench_dataset" / kind / lib
        if not path.is_dir():
            raise FileNotFoundError(path)
        for file in sorted(path.rglob("*.ipynb")):
            notebook = json.loads(file.read_text())
            entries.append(dict(task_id=str(file.relative_to(root / "data/cibench_dataset")),
                                sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                                questions=sum(c["cell_type"] == "markdown" for c in notebook["cells"])))
    protocol = dict(benchmark="OpenCompass CIBench", mode="end-to-end / infer_every; no oracle history",
                    tasks=len(entries), questions=sum(e["questions"] for e in entries),
                    entries=entries, scope_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest(),
                    upstream_cibench="5232d8cb145d2addc1c38d840b851f65368c424d",
                    upstream_opencompass="6288ea0db61150b6c2c1e37c01808fef66748e8c",
                    upstream_lagent="2e469d7dd524b18dade138a54386c91b56ff5795",
                    max_turn=3, output_tokens_per_call=512, python_call_timeout=120,
                    http_timeout=180, http_retries=0, temperature=0, top_p=1,
                    reasoning=False, compaction=False, context=262144, seed=42,
                    tool_contract_sha256=hashlib.sha256((root/'configs/tools.json').read_bytes()).hexdigest(),
                    scoring="Official CIBench metric operators; current output-directory API repair only",
                    edition=edition,
                    kernel_image=(root/'environment/IMAGE_ID').read_text().strip(),
                    execution_isolation="Fresh Docker/Jupyter kernel per notebook; public input files only",
                    actual_iopub_idle_timeout=10,
                    output_trim_chars=1024, notebook_state="persistent within notebook, reset between notebooks",
                    adapter_sha256=hashlib.sha256(Path(__file__).with_name('cibench_adapter.py').read_bytes()).hexdigest(),
                    isolated_interpreter_sha256=hashlib.sha256(Path(__file__).with_name('isolated_kernel.py').read_bytes()).hexdigest(),
                    native_error_mapping="Harness isError=true -> ActionStatusCode.API_ERROR; raw error feedback retained",
                    notes=["Function schemas replace ReAct textual serialization equally for every model.",
                           "Official CIReAct stops a question after first successful action; max 3 on failure.",
                           "Official all-prompt scope, including Chinese and visualization; oracle modes excluded."])
    with (root / ("configs/evaluation-protocol-"+edition+".json")).open('x') as handle:
        handle.write(json.dumps(protocol,indent=2) + "\n")
    base = '''from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import CIBenchDataset, CIBenchEvaluator
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from cibench_adapter import NativeCodeAgent
ROOT = ROOT_VALUE
datasets = []
SCOPES = SCOPES_VALUE
for kind, lib in SCOPES:
    datasets.append(dict(abbr=kind+"/"+lib, type=CIBenchDataset,
        path=ROOT+"/data/cibench_dataset/"+kind+"/"+lib, internet_check=False,
        reader_cfg=dict(input_columns=["questions"],output_column="references",train_split="test",test_split="test"),
        infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template="{questions}"),
                       retriever=dict(type=ZeroRetriever),inferencer=dict(type=AgentInferencer,infer_mode="every")),
        eval_cfg=dict(evaluator=dict(type=CIBenchEvaluator,user_data_dir=ROOT+"/data/cibench_dataset/datasources"),pred_role="BOT")))
models = [dict(abbr=LABEL_VALUE,type=NativeCodeAgent,root=ROOT,
    endpoint="http://127.0.0.1:18641",model=MODEL_VALUE,
    user_data_dir=ROOT+"/data/cibench_dataset/datasources",batch_size=1,
    run_cfg=dict(num_gpus=0,num_procs=1))]
work_dir = ROOT+"/evaluation/"+LABEL_VALUE+"-"+EDITION_VALUE
infer = dict(partitioner=dict(type=SizePartitioner,max_task_size=1000,strategy="split"),
             runner=dict(type=LocalRunner,max_num_workers=1,task=dict(type=OpenICLInferTask)))
'''
    for label in ["m0", "epoch1", "epoch2"]:
        text = base.replace("ROOT_VALUE",repr(str(root))).replace("SCOPES_VALUE",repr(scopes))
        text = text.replace("LABEL_VALUE",repr(label)).replace("MODEL_VALUE",repr("qwen35-ci-" + label))
        text = text.replace("EDITION_VALUE",repr(edition))
        with (root / "configs" / ("cibench_"+label+"_"+edition+".py")).open('x') as handle:
            handle.write(text)
    print(json.dumps({k:v for k,v in protocol.items() if k!='entries'}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--edition',required=True)
    args=parser.parse_args()
    main(args.root,args.edition)
