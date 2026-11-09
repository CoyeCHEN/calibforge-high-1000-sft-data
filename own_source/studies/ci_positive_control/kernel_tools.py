"""Executed only inside the isolated IPython kernel; reuse six real tools."""
import os
import sys
sys.path.insert(0,'/ci-runtime/open_harness_ai.whl')
from foundation.jobs import BashJobs
from rsi_modules.capabilities.tools import createCodingTools
from rsi_modules.capabilities.tools.bash_job import createBashJobTool
from rsi_modules.capabilities.tools.finish import createFinishTool

jobs=BashJobs('/tmp/ci-background-jobs',maxConcurrent=2)
tools=createCodingTools(os.getcwd(),{'bash':{'jobs':jobs,'timeout':120}})
tools += [createBashJobTool(jobs),createFinishTool('Check the requested result before completing.',jobs)]
registry={tool['name']:tool for tool in tools}


async def execute(name,args):
    from jsonschema import validate
    tool=registry[name]
    validate(args,tool['parameters'])
    return await tool['execute']('ci-eval',args)
