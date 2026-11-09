"""Training boundary around the existing independent Recorder-v2 reader, not Harness code."""
import hashlib
import json
import re
from pathlib import Path
import sys
from sft import qwen35 as codec
from sft.runtime.checkpoint_archive import sha256
ROOT=Path(__file__).resolve().parent
from adapters.openharness_v2 import _raw_rows, _closure_issues
SENSITIVE=re.compile(r'\b(?:sk|hf|ghp|github_pat)[_-][A-Za-z0-9_-]{12,}|authorization\s*[:=]\s*bearer\s+\S+',re.I)
PRIVATE=re.compile(r'/tests/(?:test_outputs\.py|test\.sh)|/solution/solve\.(?:sh|py)|/logs/verifier/')
SMOKE_TASKS={'harbor/hello-workdir','harbor/hello-skills','harbor/hello-healthcheck','harbor/runtime-skill-merge','harbor/extra-instruction-path'}


def checked_file(path,expected):
    p=Path(path)
    actual=sha256(p)
    if actual!=expected:raise ValueError('Source identity mismatch: '+str(p))
    return p


def verify_recorders(entry,manifest,calls,tools):
    sessions={s['session_id']for s in manifest.get('sessions',[manifest])};raw_ids=set();raw_calls=[];raw_tools=[]
    for source in entry['raw_recorders']:
        directory=Path(source['path'])
        raw_manifest=json.loads(checked_file(directory/'manifest.json',source['manifest_sha256']).read_text())
        checked_file(directory/'events.jsonl',source['events_sha256'])
        raw_ids.add(raw_manifest['session_id'])
        _,rc,rt,rr,ri=_raw_rows(directory)
        if ri['issues']or _closure_issues(rr,rc,rt):raise ValueError('Original Recorder not closed')
        raw_calls.extend(rc);raw_tools.extend(rt)
    if raw_ids!=sessions:raise ValueError('All exported session identities require original Recorder evidence')
    def projection(rows,key,fields):
        result={r[key]:codec.digest({field:r.get(field)for field in fields})for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate Recorder identity')
        return result
    fields=('session_id','run_id','payload','response','status')
    if projection(raw_calls,'call_id',fields)!=projection(calls,'call_id',fields):
        raise ValueError('Export differs from original Recorder calls or actual provider payloads')
    fields=('call_id','tool_call_id','result','raw_result','status')
    if projection(raw_tools,'execution_id',fields)!=projection(tools,'execution_id',fields):
        raise ValueError('Export differs from original Recorder tool results')
    return sorted(raw_ids)


def formal_identity(entry,result):
    if not entry.get('collection_manifest')or not entry.get('collection_manifest_sha256'):
        raise ValueError('Frozen independent teacher collection manifest required')
    identity=json.loads(checked_file(entry['collection_manifest'],entry['collection_manifest_sha256']).read_text())
    published=entry['origin']['kind']=='published_harbor_task'
    expected_kind='published_harbor_teacher_collection'if published else'm2_independent_teacher_collection'
    if identity.get('kind')!=expected_kind:raise ValueError('Unexpected teacher collection identity')
    for name,size in [('harness_commit',40),('harbor_commit',40),('harness_wheel_sha256',64)]:
        if not re.fullmatch('[0-9a-f]{'+str(size)+'}',identity.get(name,'')):raise ValueError('Missing frozen '+name)
    teacher=identity.get('teacher_model','')
    if not teacher.startswith('deepseek/')or not identity.get('teacher_model_version_basis'):raise ValueError('DeepSeek teacher identity/version basis required')
    if published:
        from .published import collection_scope
        expected=collection_scope(entry,identity)
    else:
        versions=json.loads((ROOT/'versions.json').read_text())
        expected=dict(harness_commit=versions['openharness_commit'],harbor_commit=versions['harbor_commit'])
    if any(identity[key]!=value for key,value in expected.items()):
        raise ValueError('Harness/Harbor commit differs from audited contract')
    agent=result['config']['agent']
    if agent.get('kwargs',{}).get('profile','native-cli-v1')!='native-cli-v1'or not agent.get('kwargs',{}).get('record_trajectory'):
        raise ValueError('Native Harbor profile with Recorder required')
    if agent.get('name')!='open-harness'and agent.get('import_path')!='harbor.agents.installed.open_harness:OpenHarnessAgent':
        raise ValueError('Native Harbor OpenHarnessAgent required')
    if agent['model_name']!=teacher or agent['kwargs'].get('wheel_sha256')!=identity['harness_wheel_sha256']:
        raise ValueError('Harbor teacher/wheel differs from frozen collection')
    installed=json.loads(checked_file(entry['installed_artifact'],entry['installed_artifact_sha256']).read_text())
    if installed['wheel_sha256']!=identity['harness_wheel_sha256']:raise ValueError('Actually installed Harness wheel mismatch')
    if installed.get('profile')!='native-cli-v1'or installed.get('cli_contract')!='openharness-cli-v1':
        raise ValueError('Actually installed Harness profile/CLI contract mismatch')
    approved={t['task_id']:t for t in identity.get('approved_tasks',[])}
    task=approved.get(entry['task_id'],{})
    if any(task.get(k)!=entry[k]for k in ('task_id','task_group_id','split')):raise ValueError('Task/group/split not in frozen approved collection')
    if published:
        from .published import source_files_sha256
        files_digest=source_files_sha256(entry['origin']['source_files'])
    else:files_digest=codec.digest(entry['origin']['source_files'])
    if task.get('source_files_sha256')!=files_digest:raise ValueError('Task source files differ from frozen approved task')
    if identity.get('agent_options_sha256')!=codec.digest(agent['kwargs']):raise ValueError('Native teacher agent options differ from frozen collection')
    return {k:identity[k]for k in ['teacher_model','teacher_model_version_basis','harness_commit','harbor_commit','harness_wheel_sha256']}


def read_source(entry,*,allow_fixture=False):
    official=json.loads((ROOT/'policies/evaluation-exclusion.json').read_text())['instruction_normalized_sha256']
    task=entry['task_id'];origin=entry['origin']
    if not allow_fixture and task in SMOKE_TASKS:raise ValueError('Known integration fixture forbidden for formal M2')
    if task in official or task.startswith('terminal-bench/')or origin.get('upstream_benchmark')or origin.get('evaluation_split'):
        raise ValueError('Evaluation task forbidden, including renamed upstream benchmark sources')
    if origin['kind']not in ('authored_non_evaluation_task','published_harbor_task') and not(allow_fixture and origin['kind']=='integration_fixture'):
        raise ValueError('Independent non-evaluation task provenance required')
    if not origin.get('source_files'):
        raise ValueError('Independent task source files and hashes required')
    for item in origin['source_files']:checked_file(item['path'],item['sha256'])
    result=json.loads(checked_file(entry['harbor_result'],entry['harbor_result_sha256']).read_text())
    if result['id']!=entry['trial_id']or result['task_name']!=task or result.get('exception_info')or result.get('infrastructure_errors')or result.get('secondary_errors'):
        raise ValueError('Harbor identity/error mismatch')
    reward=(result.get('verifier_result')or{}).get('rewards',{}).get('reward')
    if not isinstance(reward,(int,float))or isinstance(reward,bool)or reward!=1:
        raise ValueError('Verified Harbor success required')
    identity=formal_identity(entry,result)if not allow_fixture else{}
    if not allow_fixture:
        trial=Path(entry['harbor_result']).resolve().parent
        sources=[entry['installed_artifact']]+[r['path']for r in entry['raw_recorders']]
        if any(not Path(p).resolve().is_relative_to(trial)for p in sources):
            raise ValueError('Original Recorder and installed wheel evidence must belong to the actual Harbor Trial')
        if origin['kind']=='published_harbor_task':
            completion=json.loads((trial/'agent/open-harness.complete.json').read_text())
            if completion.get('exit_code')!=0 or (result.get('verifier_result')or{}).get('return_code')!=0:
                raise ValueError('Normal Agent completion and Verifier exit required for published success')
    directory=Path(entry['rollout_export_dir']);checked_file(directory/'manifest.json',entry['recorder_manifest_sha256'])
    manifest,calls,tools,runs,integrity=_raw_rows(directory)
    issues=integrity["issues"]+_closure_issues(runs,calls,tools)
    if issues:raise ValueError("Recorder closure failed: "+"; ".join(issues))
    native_published=(origin['kind']=='published_harbor_task'and (directory/'events.jsonl').is_file()and manifest.get('session_id'))
    if not allow_fixture and not native_published and not manifest.get('integrity',{}).get('complete'):
        raise ValueError('Formal M2 requires a validated complete Recorder export')
    if not allow_fixture:
        identity['raw_session_ids']=verify_recorders(entry,manifest,calls,tools)
    produced=[];excluded=[]
    observations={t.get('tool_call_id')for t in tools if t.get('status')in ('completed','error')}
    for raw in sorted(calls,key=lambda c:c.get('started_at','')):
        if raw.get('source','agent')!='agent':excluded.append(dict(call_id=raw['call_id'],reason='auxiliary_'+str(raw.get('source'))));continue
        if raw.get('status')not in ('completed','stop','toolUse'):
            excluded.append(dict(call_id=raw['call_id'],reason='incomplete_'+str(raw.get('status'))));continue
        payload=raw.get('payload')
        if not isinstance(payload,dict)or not isinstance(payload.get('messages'),list):raise ValueError('Actual provider payload missing')
        if not allow_fixture and raw.get('fidelity')!='provider_payload':raise ValueError('Recorded provider-payload fidelity required; no context reconstruction fallback')
        if not allow_fixture and payload.get('model')!=identity['teacher_model'].split('/',1)[1]:raise ValueError('Actual provider request teacher model mismatch')
        if raw['response'].get('stopReason')in ('length','error','aborted'):
            excluded.append(dict(call_id=raw['call_id'],reason='incomplete_response_'+raw['response']['stopReason']));continue
        for m in payload['messages']:
            if m['role']=='user':
                instruction=codec.digest(' '.join(codec.text_content(m.get('content')).split()))
                if instruction in official.values():raise ValueError('Exact evaluation instruction match')
        response=codec.normalize_message(raw['response'])
        captured=json.dumps([payload,raw['response']],ensure_ascii=False)
        if SENSITIVE.search(captured)or PRIVATE.search(captured):
            raise ValueError('Credential/private evaluation material requires quarantine, never silent rewriting')
        if {x['id']for x in response.get('tool_calls',[])}-observations:
            raise ValueError('Target tools lack captured completed executions')
        call=dict(call_id=raw['call_id'],request_messages=payload['messages'],assistant_response=raw['response'],
                  provider_payload=payload,context_fidelity='provider_payload')
        produced.append(call)
    if not allow_fixture and any(not x['reason'].startswith('auxiliary_')for x in excluded):
        raise ValueError('Incomplete agent calls require whole-task quarantine')
    if not produced:raise ValueError('No complete agent decisions')
    return produced,dict(task_id=task,trial_id=entry['trial_id'],excluded_calls=excluded,formal_training_eligible=not allow_fixture,
        harbor_result_sha256=entry['harbor_result_sha256'],verifier_reward=1,
        recorder_manifest_sha256=entry['recorder_manifest_sha256'],**identity)
