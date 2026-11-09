"""Streaming, single-worker preparation. Raw data and historical artifacts stay untouched."""
import collections
import json
import os
from pathlib import Path
import tempfile

from . import qwen35 as codec
from .runtime.checkpoint_archive import atomic_json,sha256
from .runtime.paths import workspace,config_path

POLICIES=Path(__file__).resolve().parent/'policies'


def read_approval(path,*,purpose='formal_m2'):
    approval=json.loads(Path(path).read_text())
    if approval.get('purpose')!=purpose or not approval.get('entries'):
        raise ValueError('Formal collection has no approved task entries for '+purpose)
    groups={};trials=set()
    for entry in approval['entries']:
        group=entry['task_group_id'];split=entry['split']
        if split not in ('train','validation')or group in groups and groups[group]!=split:raise ValueError('Task group leakage')
        if entry['trial_id']in trials:raise ValueError('Duplicate trial identity')
        groups[group]=split;trials.add(entry['trial_id'])
    if set(groups.values())!={'train','validation'}:raise ValueError('Separate task groups for both splits required')
    return approval


def publish(name,generators,manifest,finalize):
    from datasets import Dataset,DatasetDict,Features,Sequence,Value
    root=workspace();destination=root/'data'/name
    if destination.exists():raise FileExistsError('Prepared data preserved')
    indexes=[root/'manifests'/f'{name}-{split}-rows.jsonl'for split in generators]
    if any(path.exists()for path in indexes):raise FileExistsError('Previous preparation indexes preserved; choose a new workspace')
    features=Features({k:Sequence(Value('int32'))for k in ('input_ids','attention_mask','labels')}|
                      {k:Sequence(Value('string'))for k in ('images','videos','audios')})
    created=[]
    try:
        for path in indexes:
            path.open('x').close();created.append(path)
        with tempfile.TemporaryDirectory(prefix='prepare-',dir=root/'data')as temporary:
            temp=Path(temporary)
            data=DatasetDict({split:Dataset.from_generator(generator,features=features,cache_dir=str(temp/'cache'),
                writer_batch_size=8,keep_in_memory=False)for split,generator in generators.items()})
            finalize()
            staged=temp/'dataset';data.save_to_disk(str(staged),max_shard_size='400MB',num_proc=1)
            manifest.update(path=str(destination),formal_training_eligible=True,
                artifacts=[dict(path=str(p.relative_to(staged)),bytes=p.stat().st_size,sha256=sha256(p))for p in sorted(staged.rglob('*'))if p.is_file()])
            os.rename(staged,destination)
        atomic_json(root/'manifests'/f'{name}.json',manifest)
    except Exception as error:
        # Failed conversion must not poison the fixed data path. Preserve only
        # indexes created by this invocation; never move an old or published set.
        if created and not destination.exists():
            failures=root/'checks/preparation-failures';failures.mkdir(parents=True,exist_ok=True)
            failed=Path(tempfile.mkdtemp(prefix=name+'-',dir=failures));preserved=[]
            for path in created:
                if path.exists():path.rename(failed/path.name);preserved.append(path.name)
            atomic_json(failed/'failure.json',dict(dataset=name,error_type=type(error).__name__,
                error=str(error)[:2000],preserved_row_indexes=preserved,formal_training_eligible=False))
        raise
    return manifest


def record_row(name,split,identity,metadata):
    # The dataset is not eligible until publish completes. Each index contains
    # IDs, hashes, target ranges and tool links, never raw teacher text/secrets.
    path=workspace()/'manifests'/f'{name}-{split}-rows.jsonl'
    with path.open('a')as stream:stream.write(json.dumps(identity|metadata)+'\n')


def prepare_m1(train,validation):
    import pyarrow.parquet as pq
    policy=json.loads((POLICIES/'m1-quality.json').read_text())
    paths=dict(train=Path(train),validation=Path(validation))
    for split,path in paths.items():
        source='holdout'if split=='validation'else split
        if sha256(path)!=policy['source_sha256'][source]:raise ValueError('Pinned M1 source SHA mismatch')
    tokenizer=codec.load_tokenizer();groups=collections.defaultdict(set);seen=set();counts={};rejected=[]
    excluded={(r['split'],r['trajectory']):r['reason']for r in policy['exclusions']}
    official=set(json.loads((POLICIES/'evaluation-exclusion.json').read_text())['instruction_normalized_sha256'].values())
    def generator(split):
        def rows():
            counter=collections.Counter()
            for batch in pq.ParquetFile(paths[split]).iter_batches(batch_size=8):
                for raw in batch.to_pylist():
                    counter['source_trajectories']+=1;reason=excluded.get((split,raw['trajectory_id']))
                    if reason:
                        rejected.append(dict(split=split,trajectory=raw['trajectory_id'],reason=reason));continue
                    messages=raw['messages'];first=next(codec.text_content(m.get('content'))for m in messages if m['role']=='user')
                    if codec.digest(' '.join(first.split()))in official:raise ValueError('Evaluation instruction overlap')
                    last=max(i for i,m in enumerate(messages)if m['role']=='assistant')
                    row,meta=codec.encode(tokenizer,messages[:last+1],scope='all_assistant',cutoff=65536)
                    if meta['input_sha256']in seen:raise ValueError('Duplicate encoded M1 sample')
                    seen.add(meta['input_sha256']);groups[split].add(raw['task_group_id'])
                    counter['rows']+=1;counter['supervised_tokens']+=meta['loss_tokens'];counter['input_tokens']+=meta['text_tokens']
                    record_row('m1-64k',split,dict(trajectory=raw['trajectory_id'],task_group=raw['task_group_id'],split=split),meta)
                    yield row
            counts[split]=dict(counter)
        return rows
    def finalize():
        if groups['train']&groups['validation']:raise ValueError('M1 task-group leakage')
        for split in paths:
            if counts[split]['rows']!=policy['expected_rows'][split]or counts[split]['supervised_tokens']!=policy['expected_supervised_tokens'][split]:
                raise ValueError('Frozen M1 counts or supervision changed')
    manifest=dict(kind='M1_public_agent_sft',model_revision=codec.REVISION,template_sha256=codec.TEMPLATE_SHA,
        cutoff=65536,dataset=policy['dataset'],dataset_revision=policy['revision'],source_sha256=policy['source_sha256'],
        no_truncation=True,task_groups_disjoint=True,original_split_preserved=True,supervision_scope='all_assistant',counts=counts,exclusions=rejected)
    return publish('m1-64k',{s:generator(s)for s in paths},manifest,finalize)


def prepare_m2(approval_path,length):
    approval=read_approval(approval_path)  # Refuse zero data BEFORE tokenizer/import/download.
    contract_path=workspace()/'configs/m2-tool-contract.json'
    contract=json.loads(contract_path.read_text())
    if contract.get('status')!='APPROVED'or not contract.get('schema_sha256'):raise ValueError('M2 tool contract undecided')
    from .sources import read_source
    tokenizer=codec.load_tokenizer();counts={};rejected=[];lengths=[];tasks=[];identities=[];seen=set()
    def generator(split):
        def rows():
            counter=collections.Counter()
            for entry in approval['entries']:
                if entry['split']!=split:continue
                counter['source_trajectories']+=1
                try:calls,identity=read_source(entry)
                except(ValueError,KeyError,OSError)as error:
                    counter['quality_rejected_trajectories']+=1
                    rejected.append(dict(task=entry['task_id'],reason=str(error)));continue
                identities.append(identity);kept=0
                for call in calls:
                    counter['decisions_considered']+=1
                    try:row,meta=codec.encode_call(tokenizer,call,cutoff=2**31-1)
                    except(ValueError,AssertionError)as error:
                        counter['encoding_rejected_decisions']+=1;rejected.append(dict(task=entry['task_id'],call=call['call_id'],reason=str(error)));continue
                    if meta['tools_sha256']not in contract['schema_sha256']:
                        counter['tool_schema_rejected_decisions']+=1;rejected.append(dict(task=entry['task_id'],call=call['call_id'],reason='TOOL_SCHEMA_NOT_APPROVED'));continue
                    lengths.append(dict(trial=entry['trial_id'],tokens=meta['text_tokens'],loss_tokens=meta['loss_tokens']))
                    if meta['text_tokens']>length-66:
                        counter['overlength_rejected_decisions']+=1;rejected.append(dict(task=entry['task_id'],call=call['call_id'],reason='EXCEEDS_CUTOFF_NO_TRUNCATION'));continue
                    if meta['input_sha256']in seen:
                        counter['duplicate_rejected_decisions']+=1
                        rejected.append(dict(task=entry['task_id'],call=call['call_id'],reason='DUPLICATE_ENCODED_SEQUENCE'));continue
                    seen.add(meta['input_sha256']);counter['rows']+=1;counter['supervised_tokens']+=meta['loss_tokens'];counter['input_tokens']+=meta['text_tokens'];kept+=1
                    record_row('m2-'+str(length//1024)+'k',split,dict(task_id=entry['task_id'],task_group=entry['task_group_id'],trial_id=entry['trial_id'],call_id=call['call_id'],split=split),meta)
                    yield row
                tasks.append(dict(trial=entry['trial_id'],task_group=entry['task_group_id'],decisions=len(calls),retained=kept))
            counts[split]=dict(counter)
            if not counter['rows']:raise ValueError('No approved samples retained in '+split)
        return rows
    coverage={}
    def finalize():
        for cutoff in (98304,131072):
            fit=[x for x in lengths if x['tokens']<=cutoff-66]
            complete={t['trial']for t in tasks if len([x for x in fit if x['trial']==t['trial']])==t['decisions']}
            groups={t['task_group']for t in tasks}
            coverage[str(cutoff)]=dict(complete_trajectories=len(complete),quality_accepted_trajectories=len(tasks),
                task_groups=len(groups),task_groups_with_a_complete_trajectory=sum(any(t['trial']in complete for t in tasks if t['task_group']==g)for g in groups),
                decisions=len(fit),encodable_decisions=len(lengths),
                loss_tokens=sum(x['loss_tokens']for x in fit),encodable_loss_tokens=sum(x['loss_tokens']for x in lengths),
                basis='Length-only coverage before dedup; whole decisions, 66-token LF overhead. Quality and encoding rejections counted separately.')
    manifest=dict(kind='formal_m2',tool_contract_sha256=sha256(contract_path),model_revision=codec.REVISION,template_sha256=codec.TEMPLATE_SHA,cutoff=length,
        text_budget=length-66,source_manifest_sha256=sha256(approval_path),no_truncation=True,task_groups_disjoint=True,
        supervision_scope='current_assistant',counts=counts,exclusions=rejected,identities=identities,length_coverage=coverage)
    try:return publish('m2-'+str(length//1024)+'k',{s:generator(s)for s in ('train','validation')},manifest,finalize)
    finally:atomic_json(workspace()/'checks'/f'm2-{length//1024}k-quality.json',dict(counts=counts,exclusions=rejected))


def prepare_calibforge(approval_path,tool_contract_path):
    """One exact-history multi-turn row per successful public teacher trajectory."""
    approval=read_approval(approval_path,purpose='formal_calibforge')
    contract=json.loads(Path(tool_contract_path).read_text())
    if contract.get('status')!='APPROVED'or not contract.get('schema_sha256')or contract.get('finish')is not False:
        raise ValueError('Frozen tools and Finish-off contract required')
    for entry in approval['entries']:
        origin=entry['origin']
        if (origin.get('kind')!='published_harbor_task'or origin.get('dataset')!='AweAI-Team/CalibForge'
            or origin.get('revision')!='c2fae4c1f536461142b2a89a8b9423a8bc376088'):
            raise ValueError('Fixed public CalibForge provenance required')
    from .sources import read_source
    tokenizer=codec.load_tokenizer();counts={};rejected=[];seen=set();name='calibforge-128k'
    # The pinned native LF collator appends 66 masked image-placeholder positions
    # to text-only rows. Its alignment is 8 and 131072 is divisible by 8.
    # Reject excess real text; never shorten either inputs or targets.
    text_budget=131072-66
    def generator(split):
        def rows():
            counter=collections.Counter()
            for entry in approval['entries']:
                if entry['split']!=split:continue
                counter['source_trajectories']+=1
                try:
                    calls,identity=read_source(entry)
                    row,meta=codec.encode_multiturn_task(tokenizer,calls,cutoff=text_budget)
                    if meta['tools_sha256']not in contract['schema_sha256']:
                        raise ValueError('TOOL_SCHEMA_NOT_APPROVED')
                    if meta['input_sha256']in seen:raise ValueError('DUPLICATE_ENCODED_SEQUENCE')
                except (ValueError,KeyError,OSError,AssertionError)as error:
                    counter['rejected_trajectories']+=1
                    rejected.append(dict(task=entry['task_id'],trial=entry['trial_id'],reason=str(error)));continue
                seen.add(meta['input_sha256']);counter['rows']+=1
                counter['decisions']+=len(calls);counter['input_tokens']+=meta['text_tokens']
                counter['supervised_tokens']+=meta['loss_tokens']
                record_row(name,split,dict(task_id=entry['task_id'],task_group=entry['task_group_id'],
                    trial_id=entry['trial_id'],split=split,source_identity=identity),meta)
                yield row
            counts[split]=dict(counter)
            if not counter['rows']:raise ValueError('No approved complete multi-turn samples in '+split)
        return rows
    manifest=dict(kind='CalibForge_public_teacher_full_multiturn_sft',model_revision=codec.REVISION,
        template_sha256=codec.TEMPLATE_SHA,cutoff=131072,text_budget=text_budget,native_collator_reserved_tokens=66,
        source_manifest_path=str(Path(approval_path).resolve()),source_manifest_sha256=sha256(approval_path),
        tool_contract_sha256=sha256(tool_contract_path),codec_sha256=sha256(Path(codec.__file__)),
        no_truncation=True,task_groups_disjoint=True,supervision_scope='all_assistant_once',
        original_provider_prefixes_verified=True,row_order='Frozen approval entry order within each split',
        counts=counts,exclusions=rejected)
    def finalize():
        manifest['row_indexes']={split:dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)
            for split in ('train','validation')
            for path in [workspace()/'manifests'/f'{name}-{split}-rows.jsonl']}
    try:return publish(name,{s:generator(s)for s in ('train','validation')},manifest,finalize)
    finally:atomic_json(workspace()/'checks/calibforge-128k-quality.json',dict(counts=counts,exclusions=rejected))


def adopt_m1(path):
    """Reuse the already accepted immutable Arrow dataset; no retokenization/copy."""
    import yaml
    original=json.loads(Path(path).read_text())
    if original['model_revision']!=codec.REVISION or original['official_template_sha256']!=codec.TEMPLATE_SHA:
        raise ValueError('Existing prepared data has a different model/template')
    if not original['no_truncation']or not original['task_groups_disjoint']:raise ValueError('Existing data is not approved')
    counts={}
    for split,old in [('train','train'),('validation','holdout')]:
        counts[split]=dict(rows=original['counts'][old]['kept_trajectories'],supervised_tokens=original['counts'][old]['supervised_tokens'])
    if counts['train']['rows']!=5320 or counts['validation']['rows']!=191:raise ValueError('Frozen M1 split changed')
    manifest=dict(original,counts=counts,formal_training_eligible=True,template_sha256=codec.TEMPLATE_SHA,
                  imported_manifest_sha256=sha256(path))
    from .runtime.launch import validate_data
    validate_data(original['path'],manifest)
    config=config_path('m1-64k');cfg=yaml.safe_load(config.read_text());cfg['tokenized_path']=original['path']
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    atomic_json(workspace()/'manifests/m1-64k.json',manifest)
    return dict(reused=original['path'],counts=counts,data_copied=False)


def adopt_m2_candidates(approval_path):
    """Explicitly release frozen same-task research data; never relabel it CLEAN."""
    import yaml
    approval=json.loads(Path(approval_path).read_text())
    if approval.get('status')!='APPROVED_FOR_SAME_TASK_ADAPTATION':raise ValueError('Candidate release not approved')
    if approval.get('classification')!='TB21_SAME_TASK_ADAPTATION':raise ValueError('Use the independent source gate for CLEAN data')
    contract_path=workspace()/'configs/m2-tool-contract.json';contract=json.loads(contract_path.read_text())
    if contract.get('status')!='APPROVED'or contract.get('finish')is not False:raise ValueError('Tool contract not finalized with Finish disabled')
    source=Path(approval['candidate_manifest']);proof=Path(approval['loader_proof'])
    if sha256(source)!=approval['candidate_manifest_sha256']or sha256(proof)!=approval['loader_proof_sha256']:raise ValueError('Candidate evidence changed')
    candidate=json.loads(source.read_text());checked=json.loads(proof.read_text())
    if candidate['classification']!=approval['classification']or candidate['model_revision']!=codec.REVISION or candidate['template_sha256']!=codec.TEMPLATE_SHA:raise ValueError('Candidate identity mismatch')
    if not checked['passed']or checked['source_manifest_sha256']!=sha256(source)or not checked['task_groups_disjoint']or not checked['no_truncation']:raise ValueError('Missing complete candidate Loader proof')
    if not set(checked['tool_schema_sha256'])<=set(contract['schema_sha256']):raise ValueError('Candidate tool schemas not approved; do not rewrite historical actions')
    if not set(checked['tools_declared_task_dependence'])<=set(contract['tools']):raise ValueError('Candidate tools outside approved action space')
    path=source.parent/'tokenized';artifacts=[dict(item,path=str(Path(item['path']).relative_to('tokenized')))for item in candidate['artifacts']]
    manifest=dict(candidate,path=str(path),artifacts=artifacts,formal_training_eligible=True,
        kind='benchmark_same_task_adaptation_m2',counts={s:dict(rows=checked['counts'][s]['rows'],supervised_tokens=checked['counts'][s]['loss_tokens'])for s in ('train','validation')},
        tool_contract_sha256=sha256(contract_path),release_approval_sha256=sha256(approval_path),candidate_manifest_sha256=sha256(source),
        evaluation_restriction='Overlapping benchmark tasks are same-task adaptation, never unseen-task generalization')
    from .runtime.launch import validate_data
    validate_data(path,manifest)
    destination=workspace()/'manifests/m2-128k.json'
    if destination.exists():raise FileExistsError('Existing release preserved')
    config=config_path('m2-128k');cfg=yaml.safe_load(config.read_text());cfg['tokenized_path']=str(path)
    config.write_text(yaml.safe_dump(cfg,sort_keys=False));atomic_json(destination,manifest)
    return dict(classification=manifest['classification'],path=str(path),counts=manifest['counts'],data_copied=False)
