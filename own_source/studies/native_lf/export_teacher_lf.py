"""Export canonical messages (whole CI trace or Teacher decision) to ShareGPT.

The locked LF FunctionFormatter consumes JSON inside its native tool-call slot.
Dynamic argument dicts stay in strings, while Thinking and body stay as text.
Tokenization, history masking, padding and labels belong to LF.
"""
import argparse
import json
from pathlib import Path


def convert(record, ci_union_null_padding=False):
    messages = []
    for message in record['messages']:
        role = message['role']
        content = message.get('content') or ''
        if role == 'assistant':
            content = '<think>\n' + (message.get('reasoning_content') or '') + '\n</think>\n\n' + content
            if calls := message.get('tool_calls'):
                functions = []
                for call in calls:
                    function = dict(call['function'])
                    if isinstance(function['arguments'], str):
                        function['arguments'] = json.loads(function['arguments'])
                    if ci_union_null_padding:
                        function['arguments'] = {k: v for k, v in function['arguments'].items() if v is not None}
                    functions.append(function)
                payload = json.dumps(functions, ensure_ascii=False)
                payload = payload.replace('<', '\\u003c').replace('>', '\\u003e')
                content += '<tool_call>' + payload + '</tool_call>'
                role = 'function_call'
        elif role == 'tool':
            role = 'observation'
            if messages and messages[-1]['from'] == 'observation':
                messages[-1]['value'] += '\n</tool_response>\n<tool_response>\n' + content
                continue
        elif role == 'system':
            content += '\n'
        tag = {'user': 'human', 'assistant': 'gpt'}.get(role, role)
        messages.append({'from': tag, 'value': content})
    tools = record['tools']
    return {'conversations': messages, 'tools': tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='JSONL: messages history plus current assistant, and tools')
    parser.add_argument('output', type=Path, help='New dataset directory')
    parser.add_argument('--ci-union-null-padding', action='store_true',
                        help='Only for published CI union columns; omit absent argument keys padded with null')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.input.open() as source, (args.output / 'data.jsonl').open('x') as target:
        for line in source:
            target.write(json.dumps(convert(json.loads(line), args.ci_union_null_padding), ensure_ascii=False) + '\n')
    (args.output / 'dataset_info.json').write_text(json.dumps({'real': {
        'file_name': 'data.jsonl', 'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', 'tools': 'tools'},
    }}, indent=2) + '\n')


if __name__ == '__main__':
    main()
