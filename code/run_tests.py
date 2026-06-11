
import json
from pathlib import Path
from llm import LLM
from benchmark_runner import run_benchmark
from os import environ



RESULTS_ROOT = Path('/home/dario/miniconda3/envs/denv/Personas-Effect-On-Factual-Knowledge-Retrieval/results')
BENCHMARK_ORDER = ['aime', 'mmlu_pro', 'mmlu', 'gpqa']
SHOT_MODES = ['zero-shot', 'few-shot']
PERSONAS_OPTIONS = [False, True]
TEST_RUNS = 5

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

def sanitize_name(value: str) -> str:
    return value.replace('/', '_').replace(':', '_').replace(' ', '_')

def get_reasoning_variants(model_id: str, reasoning_supported: bool) -> list:
    if 'gpt-oss' in model_id.lower():
        return ['low', 'medium', 'high']
    
    if model_id == 'Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ':
        return [True]
    
    return [False, True] if reasoning_supported else [False]

def make_result_filename(model_id: str, reasoning_value, personas: bool, shot_mode: str) -> str:
    reasoning_tag = str(reasoning_value).lower()
    personas_tag = str(personas).lower()
    return f'reasoning_{reasoning_tag}_personas_{personas_tag}_{shot_mode}.json'


def get_output_path(model_id: str, benchmark_dir: Path, reasoning_value, personas: bool, shot_mode: str) -> Path:
    model_tag = sanitize_name(model_id)
    return benchmark_dir / model_tag / make_result_filename(model_id, reasoning_value, personas, shot_mode)


def json_default(obj):
    if hasattr(obj, 'dict') and callable(obj.dict):
        return obj.dict()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def write_results(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, default=json_default)


def apply_reasoning(model: LLM, model_id: str, reasoning_value) -> None:
    if 'gpt-oss' in model_id.lower():
        model.set_gpt_oss_reasoning_effort(reasoning_value)
    else:
        model.set_reasoning(bool(reasoning_value))


if __name__ == '__main__':


    models_list = [
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", [True]], #running on 3
        ["meta-llama/Llama-3.1-8B-Instruct", [False]], # DONE
        ["kosbu/Llama-3.3-70B-Instruct-AWQ", [False]], # running on 0
        ["Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ", [True]], # running on 1
        ["openai/gpt-oss-20b", ['low', 'medium', 'high']], # running on 2
        ["Qwen/Qwen3.5-35B-A3B-FP8", [True, False]], # running on 4
        ["Qwen/Qwen3.5-9B", [True, False]] # running on 5
    ]

    environ['CUDA_VISIBLE_DEVICES'] = '5'

    index = 6

    model_id = models_list[index][0]
    reasoning_variants = models_list[index][1]

    model_lower = model_id.lower()
    configurations = []

    for personas in PERSONAS_OPTIONS:
        for shot_mode in SHOT_MODES:
            configs = {'personas': personas, 'shot_mode': shot_mode}
            configurations.append(configs)

    model = LLM(model_id=model_id, reasoning= True in reasoning_variants or len(reasoning_variants) == 3)

    print(f'Running model: {model_id} with {len(configurations)} dataset configs and {len(reasoning_variants)} reasoning variants.')

    for benchmark_label in BENCHMARK_ORDER:
        benchmark_dir = RESULTS_ROOT / benchmark_label
        benchmark_dir.mkdir(parents=True, exist_ok=True)


        for config in configurations:
            personas = config['personas']
            shot_mode = config['shot_mode']

            for reasoning_value in reasoning_variants:
                output_path = get_output_path(model_id, benchmark_dir, reasoning_value, personas, shot_mode)
                if output_path.exists():
                    print(f'Skipping existing file: {output_path.name}')
                    continue
                apply_reasoning(model, model_id, reasoning_value)
                print(f'Running {benchmark_label} | personas={personas} | shot_mode={shot_mode} | reasoning={reasoning_value}')

                run_results = []
                for attempt in range(1, TEST_RUNS + 1):
                    print(f'  Attempt {attempt}/{TEST_RUNS}')
                    result = run_benchmark(model, benchmark_label=benchmark_label, shot_mode=shot_mode, personas=personas)
                    run_results.append(result)

                write_results(output_path, run_results)
                print(f'  Saved {output_path}')

    print(f'Completed model {model_id}')