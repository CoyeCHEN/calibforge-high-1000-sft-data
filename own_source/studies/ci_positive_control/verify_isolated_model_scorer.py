"""Known-correct model execution and full official evaluator, outside formal scope."""
import base64
import json
from pathlib import Path
import re
import sys

from cibench_adapter import NativeCodeAgent
from opencompass.datasets.cibench import CIBenchEvaluator


def main(root):
    root=Path(root)
    data=str(root/'data/cibench_dataset/datasources')
    a=NativeCodeAgent(str(root),'http://127.0.0.1:18641','qwen35-ci-m0',data)
    try:
        _,first,_=a.chat('Use Python to set ci_transport_state = 40 and print it.')
        second,steps,_=a.chat('Using ci_transport_state from the previous execution, print ci_transport_state + 2.')
        assert '42' in second and all(s['type']=='IPythonInterpreter' and s['state']==0 for s in first+steps)
        evaluator=CIBenchEvaluator(user_data_dir=data)
        assert evaluator.correct_step(steps,'42')['score'] is True
        assert evaluator.correct_step(steps,'100')['score'] is False
        image=a.python('import matplotlib.pyplot as plt\nplt.plot([0,1],[0,1]); plt.show()',timeout=30)
        png=Path(re.search(r'!\[fig-[0-9]*\]\((.*?)\)',image.result['image_path']).group(1))
        score=evaluator.vis_similarity_step([dict(type=image.type,args=image.args,result=image.result,errmsg=image.errmsg)],
            base64.b64encode(png.read_bytes()).decode(),'non-formal identity')['score']
        assert abs(score-1)<1e-6
        evaluator._out_dir=str(root/'checks/known-correct-official-score')
        full=evaluator.score(predictions=[second],references=[dict(outputs=['42'],tags=['general'],metadata={})],
            steps=[steps and [steps]],origin_prompt=[['Print forty two.']])
        assert full['general_correct']==100. and full['executable']==100.,full
        result=dict(status='PASSED',first=first,second=steps,real_m0_generation=True,stateful_real_python=True,
            official_numeric_positive_negative=True,official_vis_identity=score,official_full_score=full)
    finally:
        a.reset()
    (root/'checks/isolated-model-and-scorer.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ISOLATED_REAL_MODEL_AND_FULL_OFFICIAL_SCORER_PASSED',flush=True)


if __name__=='__main__':main(sys.argv[1])
