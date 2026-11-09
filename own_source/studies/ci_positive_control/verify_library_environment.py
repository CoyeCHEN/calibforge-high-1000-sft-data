"""Real isolated CPU library/asset smoke, without benchmark answers."""
import json
import os
from pathlib import Path
import sys
from isolated_kernel import IsolatedIPython


def main(root):
    root=Path(root);os.environ['CI_ROOT']=str(root)
    interpreter=IsolatedIPython(name='IPythonInterpreter',user_data_dir=str(root/'data/cibench_dataset/datasources'))
    code='''
import json, numpy as np, pandas as pd, scipy, cv2, sklearn, seaborn, lightgbm
import torch, torchvision, tensorflow as tf, nltk
for asset in ['punkt', 'punkt_tab', 'stopwords']:
    assert nltk.download(asset, quiet=True, raise_on_error=True)
assert nltk.word_tokenize('Two cells share state.')[-1] == '.'
assert 'the' in nltk.corpus.stopwords.words('english')
assert int(torch.tensor([1, 2]).sum()) == 3
assert int(tf.reduce_sum(tf.constant([1, 2]))) == 3
assert cv2.cvtColor(np.zeros((2,2,3),dtype=np.uint8),cv2.COLOR_BGR2GRAY).shape == (2,2)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
x,y=load_iris(return_X_y=True)
assert DecisionTreeClassifier(random_state=42).fit(x,y).score(x,y) == 1
print('ALL_TEN_LIBRARIES_AND_NLTK_ASSETS_PASSED')
'''
    try:
        result=interpreter(code,timeout=120)
        assert int(result.state)==0 and 'ALL_TEN_LIBRARIES_AND_NLTK_ASSETS_PASSED' in result.result['text'],result
        proof=dict(status='PASSED',real_container=True,gpu_in_sandbox=False,result=result.result,
                   scope='Imports all ten benchmark libraries, real NLTK public downloads and representative computations')
    finally:interpreter.reset()
    (root/'checks/library-environment.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('LIBRARY_ENVIRONMENT_PASSED',flush=True)


if __name__=='__main__':main(sys.argv[1])
