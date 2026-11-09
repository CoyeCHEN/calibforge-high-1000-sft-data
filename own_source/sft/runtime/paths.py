"""One explicit runtime workspace, independent of the source checkout."""
import json
import os
from pathlib import Path

PUBLIC_SOURCES={
    'new-cf416-m1':('AweAI-Team/CalibForge','c2fae4c1f536461142b2a89a8b9423a8bc376088',416),
    'new-cf416-tmax356-m2':('allenai/TMax-15K','e3ded940596cc611c9815684ba51a7e8ee0cffc9',356)}
PUBLIC_STAGES=tuple(PUBLIC_SOURCES)
STAGES=('m1-64k','m2-128k','m2-96k','m3-a-128k','c1-b-128k')+PUBLIC_STAGES
CALIBFORGE_STAGES=('m3-a-128k','c1-b-128k')

def workspace():
    return Path(os.environ['OPENHARNESS_SFT_WORKSPACE']).resolve()


def runtime_config():
    return json.loads((workspace()/'configs/runtime.json').read_text())


def config_path(stage):
    if stage not in STAGES:
        raise ValueError('Unknown stage')
    return workspace()/'configs'/f'{stage}.yaml'


def data_manifest_path(stage):
    config_path(stage)  # Validate the same public stage identity before path use.
    name='calibforge-128k'if stage in CALIBFORGE_STAGES else stage
    return workspace()/'manifests'/f'{name}.json'
