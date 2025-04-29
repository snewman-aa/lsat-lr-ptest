import json

def fix_json():
    with open('data/lr_explanations.jsonl') as f:
        explanations = [json.loads(line) for line in f]
        # Switch values for answers and explanation
        for explanation in explanations:
            explanation['answers'], explanation['explanation'] = explanation['explanation'], explanation['answers']
        with open('data/lr_explanations_fixed.jsonl', 'w') as new_file:
            for explanation in explanations:
                new_file.write(json.dumps(explanation, ensure_ascii=False) + '\n')


def convert_json_to_tsv():
    with open('data/lr_explanations_fixed.jsonl') as f:
        questions = [json.loads(line) for line in f]
        # Replace tabs in each value with spaces to avoid issues with TSV format
        for question in questions:
            for key, value in question.items():
                if isinstance(value, str):
                    question[key] = value.replace('\t', ' ')
        # Create TSV, with independent A, B, C, D, E columns for answers
        with open('data/lsat_questions.tsv', 'w') as new_file:
            # Write header
            new_file.write('question_number\tstimulus\tprompt\tA\tB\tC\tD\tE\tcorrect_answer\texplanation\n')
            for question in questions:
                # Write each question to the TSV file
                new_file.write(f"{question['question_number']}\t{question['stimulus']}\t{question['prompt']}\t"
                               f"{'\t'.join(question['answers'])}\t{question['correct_answer']}\t{question['explanation']}\n")
        print("Converted JSONL to TSV successfully.")


if __name__ == '__main__':
    convert_json_to_tsv()