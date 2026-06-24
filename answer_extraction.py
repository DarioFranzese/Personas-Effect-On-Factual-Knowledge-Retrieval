import json
from pathlib import Path
from evaluator import AnswerExtractor
from math import ceil

extractor = AnswerExtractor()

results_base = Path("/home/dario/miniconda3/envs/denv/Personas-Effect-On-Factual-Knowledge-Retrieval/results")
answers_base = Path("/home/dario/miniconda3/envs/denv/Personas-Effect-On-Factual-Knowledge-Retrieval/answers")

for input_path in results_base.glob("*/*/*.json"):
    dataset = input_path.parts[-3]
    model = input_path.parts[-2]
    filename = input_path.name

    output_path = answers_base / dataset / model / filename
    

    if output_path.exists():
        continue

    print(output_path)
    answers = []


    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as f:
        results = json.load(f)

    runs = []
    for run_results in results:

        items = []
        for item in run_results:

            output = item.get("output", {})
            output_text = output.get("output_text") or output.get("text", "")
            finish_reason = output.get("finish_reason")   
            golden = item.get("answer")
            question = item.get("question") or item.get("problem") or item.get('Question')
            subject = item.get("subject") or item.get("category")
            logprobs = output.get("logprobs")

            extractor_used, extracted = extractor.extract(
                output_text=output_text,
                question= question,
                dataset=dataset,
                golden = golden,
                finish_reason=finish_reason,
            )

            answers.append(extracted)

            try:
                answer_logprob, cleaned_logprobs = extractor.get_logprobs(extracted, logprobs)
            except Exception:
                print(f'Extractor used: {extractor_used} Output text: {output_text}')
                raise Exception

            items.append({
                "subject": subject,
                "question": question,
                "golden_label": golden,
                "finish_reason": finish_reason,
                "extracted_answer": extracted,
                "extracted_logprob": answer_logprob,
                'logprobs': cleaned_logprobs,
            })

        runs.append(items)
    
    print(answers)
    answers = [] 

    output_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n")