"""Freeze approved teacher scope and admit verified trials; Harbor owns execution."""
import json
from pathlib import Path

from . import qwen35 as codec
from .runtime.checkpoint_archive import atomic_json,sha256
from .runtime.paths import workspace
from .sources import SMOKE_TASKS,SENSITIVE,read_source


def freeze(scope_path,base_job,wheel,teacher,version_basis):
    root=workspace();scope=json.loads(Path(scope_path).read_text());job=json.loads(Path(base_job).read_text())
    versions=json.loads((Path(__file__).parent/'versions.json').read_text())
    published=scope.get('kind')=='approved_published_harbor_tasks'
    if published:
        from .published import approved_scope
        execution=approved_scope(scope)
        if execution['harness_wheel_sha256']!=sha256(wheel):raise ValueError('Wheel differs from approved public collection contract')
    else:
        if scope.get('kind')!='approved_independent_tasks'or not scope.get('tasks'):raise ValueError('Explicit approved non-evaluation task scope required')
        execution=dict(harness_commit=versions['openharness_commit'],harbor_commit=versions['harbor_commit'],harness_wheel_sha256=sha256(wheel))
    if not teacher.startswith('deepseek/')or not version_basis:raise ValueError('Actual DeepSeek model ID and version basis required')
    if len(job.get('agents',[]))!=1:raise ValueError('One explicit native teacher agent required')
    agent=job['agents'][0];options=agent['kwargs']
    if options.get('profile','native-cli-v1')!='native-cli-v1':raise ValueError('Legacy Harbor profile is not supported for new collection')
    options.update(profile='native-cli-v1',record_trajectory=True,wheel_path=str(Path(wheel).resolve()),wheel_sha256=sha256(wheel),cli_contract='openharness-cli-v1')
    agent.update(import_path='harbor.agents.installed.open_harness:OpenHarnessAgent',model_name=teacher)
    agent.pop('name',None);agent.setdefault('env',{})['DEEPSEEK_API_KEY']='${DEEPSEEK_API_KEY}'
    if SENSITIVE.search(json.dumps(job)):raise ValueError('Job config contains a literal credential')
    official=json.loads((Path(__file__).parent/'policies/evaluation-exclusion.json').read_text())['instruction_normalized_sha256']
    tasks=[];groups={}
    for entry in scope['tasks']:
        if not published and(entry.get('origin')!='authored_non_evaluation_task'or entry.get('upstream_benchmark')or entry.get('evaluation_split')):
            raise ValueError('Independent authored task provenance required; renaming evaluation tasks is forbidden')
        task=entry['task_id'];group=entry['task_group_id'];split=entry['split'];path=Path(entry['path']).resolve()
        if task in official or task in SMOKE_TASKS or task.startswith('terminal-bench/'):raise ValueError('Evaluation or integration fixture forbidden')
        if split not in ('train','validation')or group in groups and groups[group]!=split:raise ValueError('Task-group leakage')
        groups[group]=split
        instruction=codec.digest(' '.join((path/'instruction.md').read_text().split()))
        if instruction in official.values():raise ValueError('Known official evaluation instruction')
        files=[dict(path=str(p),sha256=sha256(p))for p in sorted(path.rglob('*'))if p.is_file()]
        if published:
            from .published import task_origin,source_files_sha256
            files=[dict(item,relative_path=str(Path(item['path']).relative_to(path)))for item in files]
            task_origin(entry['origin'],entry,scope)
            if files!=entry['origin']['source_files']:raise ValueError('Published task inventory differs from actual source directory')
        tasks.append(dict(task_id=task,task_group_id=group,split=split,path=str(path),instruction_sha256=instruction,
                          source_files=files,source_files_sha256=source_files_sha256(files)if published else codec.digest(files)))
    if len({t['task_id']for t in tasks})!=len(tasks):raise ValueError('Duplicate task identity')
    job.update(tasks=[dict(path=t['path'])for t in tasks],datasets=[],n_attempts=1)
    manifest=dict(kind='published_harbor_teacher_collection'if published else'm2_independent_teacher_collection',teacher_model=teacher,teacher_model_version_basis=version_basis,
        **execution,
        approved_tasks=tasks,agent_options_sha256=codec.digest(options),approved_scope_sha256=sha256(scope_path))
    if published:manifest['approved_scope']=str(Path(scope_path).resolve())
    destination=root/'manifests/teacher-collection.json'
    if destination.exists():raise FileExistsError('Existing frozen collection preserved')
    atomic_json(destination,manifest);atomic_json(root/'configs/teacher-job.json',job)
    return dict(collection=str(destination),native_job_config=str(root/'configs/teacher-job.json'),
                command=['harbor','run','--config',str(root/'configs/teacher-job.json')],tasks=len(tasks),executed=False)


def approve(candidate_path,*,purpose='formal_m2'):
    if purpose not in ('formal_m2','formal_calibforge'):raise ValueError('Unknown formal collection purpose')
    candidates=json.loads(Path(candidate_path).read_text());approved=[];rejected=[];groups={}
    for entry in candidates['entries']:
        try:
            calls,identity=read_source(entry)
            group=entry['task_group_id'];split=entry['split']
            if group in groups and groups[group]!=split:raise ValueError('Task group crosses splits')
            groups[group]=split;approved.append(entry)
        except (ValueError,KeyError,OSError)as error:
            rejected.append(dict(task=entry.get('task_id'),trial=entry.get('trial_id'),reason=str(error)))
    result=dict(approved=len(approved),rejected=rejected,quality_rejection_fraction=len(rejected)/len(candidates['entries'])if candidates['entries']else None)
    name='calibforge'if purpose=='formal_calibforge'else'm2'
    atomic_json(workspace()/'checks'/f'{name}-source-approval.json',result)
    if set(groups.values())!={'train','validation'}:raise ValueError('Verified successes in independent train and validation groups required before approval')
    destination=workspace()/'manifests'/f'{name}-approved-sources.json'
    if destination.exists()and json.loads(destination.read_text()).get('entries'):raise FileExistsError('Existing approved sources preserved')
    atomic_json(destination,dict(purpose=purpose,entries=approved))
    return result
