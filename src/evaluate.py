from transformers import pipeline
import json
from collections import Counter
from datasets import load_dataset, load_from_disk


def compute_f1(a_gold, a_pred):
	""" Compute the F1 score for predicted answer and real answer """
	gold_toks = a_gold.lower().split()
	pred_toks = a_pred.lower().split()
	common = Counter(gold_toks) & Counter(pred_toks)
	num_same = sum(common.values())
	if (num_same == 0):
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def main():
	model_checkpoint = "qa_model_final"
	question_answerer = pipeline("question-answering", model=model_checkpoint)

	corpus = json.load(open("corpus/hf_dataset_qa_pairs.json"))
	# corpus = load_dataset("CATIE-AQ/frenchQA")
	# corpus = corpus["validation"]

	counter = 0
	total_f1 = 0
	for item in corpus["data"]:
		context = item["context"]
		question = item["question"]
		answer = question_answerer(question=question, context=context)["answer"]
		if item["answers"]["text"]:
			true_answer = item["answers"]["text"][0]
		else:
			continue

		f1_score = compute_f1(true_answer, answer)
		total_f1 += f1_score
		counter += 1
		print(f"Answer: {answer}")
		print(f"\nTrue answer: {true_answer}")
		print(f"F1 Score: {f1_score:.4f}")
		print("-" * 20)

	average_f1 = total_f1 / counter
	print(f"FINAL F1 Score: {average_f1:.4f}")


if __name__ == "__main__":
	main()
