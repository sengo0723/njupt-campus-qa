"""Isolated LoRA experiments. Selection uses development/validation sets, never test80."""
from pathlib import Path
import argparse
import functools
import gc
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / 'results/tuning'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def cache(args):
    from njupt_qa.engine import Engine
    engine = Engine()
    cases = read_rows(ROOT / args.cases)
    output = []
    for case in cases:
        prepared = engine.prepare(case['question'], 'sft_rag', calculator=False, guardrails=False)
        output.append({'case': case, **prepared})
        print('cache', case['id'], flush=True)
    write_rows(ROOT / args.output, output)
    save((ROOT / args.output).with_suffix('.protocol.json'), {
        'cases_sha256': hashlib.sha256((ROOT / args.cases).read_bytes()).hexdigest(),
        'calculator': False, 'guardrails': False, 'retrieval': 'unchanged production full pipeline',
        'purpose': 'Development/validation selection; original test80 is not used here.',
    })


def evaluate(args):
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx.utils import tree_flatten
    from njupt_qa.evaluation import grade, summarize, csv_export
    from njupt_qa.dataset import SYSTEM
    out = TUNING / args.name
    out.mkdir(parents=True, exist_ok=True)
    adapter = str(ROOT / args.adapter) if args.adapter else None
    model, tokenizer = load(str(ROOT / 'models/qwen3-base-mlx'), adapter_path=adapter, trust_remote_code=False)
    if adapter and args.strength != 1:
        # Change the low-rank delta only; base weights remain frozen and unchanged.
        updates = [(name, value * args.strength) for name, value in tree_flatten(model.parameters()) if name.endswith('lora_b')]
        assert updates
        model.load_weights(updates, strict=False)
        mx.eval(model.parameters())
    cases = read_rows(ROOT / args.cache)
    generation = {'temperature': 0, 'repeat_penalty': 1.1, 'repeat_last_n': 64, 'max_tokens': args.max_tokens, 'seed': 42}
    config = {'adapter': args.adapter, 'strength': args.strength, 'runtime': 'MLX BF16 + unmerged LoRA',
              'generation': generation, 'cache_sha256': hashlib.sha256((ROOT / args.cache).read_bytes()).hexdigest()}
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == config, 'Existing run has a different configuration'
    save(out / 'protocol.json', config)
    summaries = {}
    for mode in args.modes:
        path = out / f'{mode}.jsonl'
        output = read_rows(path) if path.exists() else []
        done = {row['case']['id'] for row in output}
        for prepared in cases:
            case = prepared['case']
            if case['id'] in done:
                continue
            messages = prepared['messages'] if mode == 'sft_rag' else [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': case['question']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            start = time.monotonic()
            answer = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                              sampler=make_sampler(temp=0),
                              logits_processors=make_logits_processors(repetition_penalty=1.1, repetition_context_size=64), verbose=False).strip()
            result = {'answer': answer, 'citations': prepared['citations'] if mode == 'sft_rag' else [],
                      'elapsed_ms': (time.monotonic() - start) * 1000}
            row = {'case': case, 'question': case['question'], 'mode': mode, **result, 'grade': grade(case, result)}
            output.append(row)
            write_rows(path, output)
            print(args.name, mode, case['id'], row['grade']['correct'], round(result['elapsed_ms']), flush=True)
            mx.clear_cache()
        summaries[mode] = summarize(output)
        csv_export(out / f'{mode}.csv', output)
        save(out / 'summary.json', summaries)
    print('EVALUATION COMPLETE', args.name, {mode: v['accuracy'] for mode, v in summaries.items()}, flush=True)


def train(args):
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, load_dataset, train_model
    from train_lora import Recorder
    out = ROOT / 'models/tuning' / args.name
    assert not (out / 'metrics.jsonl').exists(), 'Refusing to overwrite a completed/partial experiment'
    out.mkdir(parents=True, exist_ok=True)
    defaults = {**CONFIG_DEFAULTS, **{k: v for k, v in vars(build_parser().parse_args([])).items() if v is not None}}
    config = {'model': str(ROOT / 'models/qwen3-base-mlx'), 'data': str(ROOT / args.data), 'train': True, 'test': False,
              'fine_tune_type': 'lora', 'optimizer': 'adamw', 'optimizer_config': {'adamw': {'weight_decay': .01}},
              'num_layers': args.layers, 'batch_size': 1, 'grad_accumulation_steps': 8, 'iters': args.iters,
              'learning_rate': args.lr, 'max_seq_length': args.max_length, 'seed': args.seed, 'mask_prompt': True,
              'grad_checkpoint': args.grad_checkpoint, 'steps_per_report': 50, 'steps_per_eval': args.save_every,
              'save_every': args.save_every, 'val_batches': -1, 'adapter_path': str(out),
              'lora_parameters': {'keys': args.keys, 'rank': args.rank, 'scale': args.scale, 'dropout': args.dropout},
              'report_to': None, 'resume_adapter_file': None, 'trust_remote_code': False}
    if args.warmup:
        updates = max(1, args.iters // 8)
        config['lr_schedule'] = {'name': 'cosine_decay', 'arguments': [args.lr, max(1, updates - args.warmup), args.lr / 10], 'warmup': args.warmup}
    train_args = argparse.Namespace(**(defaults | config))
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(train_args.model, trust_remote_code=False)
    if args.kl_beta:
        import importlib
        from lora_regularization import reference_kl_loss
        reference, _ = load(train_args.model, trust_remote_code=False)
        # MLX compilation must capture the frozen reference arrays as model state.
        # This child stays frozen and is absent from saved trainable adapters.
        model.reference_base = reference
        module = importlib.import_module('mlx_lm.lora')
        module.train = functools.partial(module.train, loss=reference_kl_loss(reference, args.kl_beta, args.kl_window))
        config.update(reference_kl_beta=args.kl_beta, reference_kl_window=args.kl_window,
                      objective='assistant-token CE + beta KL(frozen base || student), KL on last window answer positions')
        train_args.reference_kl_beta = args.kl_beta
        train_args.reference_kl_window = args.kl_window
    raw = getattr(tokenizer, '_tokenizer', tokenizer)
    raw.apply_chat_template = functools.partial(raw.apply_chat_template, enable_thinking=False)
    train_set, valid_set, _ = load_dataset(train_args, tokenizer)
    callback = Recorder(model, out)
    start = time.monotonic()
    train_model(train_args, model, train_set, valid_set, callback)
    summary = {'completed': True, 'config': config, 'samples': len(train_set), 'iterations': args.iters,
               'elapsed_seconds': time.monotonic() - start, 'peak_memory_gb': mx.get_peak_memory() / 1e9,
               'best_loss_step': callback.best_step, 'best_loss': callback.best,
               'trainable_parameters': sum(v.size for _, v in tree_flatten(model.trainable_parameters())),
               'dataset_sha256': hashlib.sha256((ROOT / args.data / 'train.jsonl').read_bytes()).hexdigest()}
    save(out / 'training_summary.json', summary)
    save(TUNING / args.name / 'training_summary.json', summary)
    print('TRAINING COMPLETE', args.name, summary, flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--cases', default='data/eval/dev.jsonl')
    c.add_argument('--output', default='data/tuning/dev_cache.jsonl')
    e = sub.add_parser('eval')
    e.add_argument('--name', required=True)
    e.add_argument('--adapter')
    e.add_argument('--strength', type=float, default=1.)
    e.add_argument('--cache', default='data/tuning/dev_cache.jsonl')
    e.add_argument('--modes', nargs='+', choices=['sft', 'sft_rag'], default=['sft_rag', 'sft'])
    e.add_argument('--max-tokens', type=int, default=768)
    t = sub.add_parser('train')
    t.add_argument('--name', required=True)
    t.add_argument('--data', default='data/sft')
    t.add_argument('--lr', type=float, default=3e-5)
    t.add_argument('--rank', type=int, default=16)
    t.add_argument('--scale', type=float, default=2.)
    t.add_argument('--dropout', type=float, default=.05)
    t.add_argument('--layers', type=int, default=28)
    t.add_argument('--keys', nargs='+', default=['self_attn.q_proj', 'self_attn.v_proj'])
    t.add_argument('--iters', type=int, default=1200)
    t.add_argument('--save-every', type=int, default=200)
    t.add_argument('--max-length', type=int, default=1536)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--warmup', type=int, default=0)
    t.add_argument('--grad-checkpoint', action='store_true')
    t.add_argument('--kl-beta', type=float, default=0.)
    t.add_argument('--kl-window', type=int, default=128)
    args = p.parse_args()
    {'cache': cache, 'eval': evaluate, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
