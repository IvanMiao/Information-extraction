from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import numpy as np
import evaluate
import collections
from tqdm.auto import tqdm

from datasets import load_dataset

dataset = load_dataset("CATIE-AQ/frenchQA")

# print elements in dataset to test if all works
# print("Context: ", dataset["train"][10]["context"])
# print("Question: ", dataset["train"][10]["question"])
# print("Answer: ", dataset["train"][10]["answers"])

model_checkpoint = "distilbert/distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_train(example):
	context = example["context"]
	question = [q.strip() for q in example["question"]]

	# configure tokenizer to fit the task
	input = tokenizer(
		question,
		context,
		max_length=384,
		truncation=True,
		stride=128,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		padding="max_length",
		)

	offset_mapping = input.pop("offset_mapping")
	sample_map = input.pop("overflow_to_sample_mapping")
	answers = example["answers"]
	start_positions = []
	end_positions = []

	for i, offset in enumerate(offset_mapping):
		sample_idx = sample_map[i]
		answer = answers[sample_idx]

		if not answer["answer_start"] or not answer["text"]:
			start_positions.append(0)
			end_positions.append(0)
			continue

		start_char = answer["answer_start"][0]
		end_char = answer["answer_start"][0] + len(answer["text"][0])
		sequence_ids = input.sequence_ids(i)

		idx = 0
		while sequence_ids[idx] != 1:
			idx += 1
		context_start = idx
		while sequence_ids[idx] == 1:
			idx += 1
		context_end = idx - 1

		# if answer in not completely in context, lable is (0, 0)
		if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
			start_positions.append(0)
			end_positions.append(0)
		else:
			# otherwise, it's the position of start_tokens and end_token
			idx = context_start
			while idx <= context_end and offset[idx][0] <= start_char:
				idx += 1
			start_positions.append(idx - 1)

			idx = context_end
			while idx >= context_start and offset[idx][1] >= end_char:
				idx -= 1
			end_positions.append(idx + 1)
	
	input["start_positions"] = start_positions
	input["end_positions"] = end_positions
	return input


def preprocess_validation_examples(example):
	context = example["context"]
	questions = [q.strip() for q in example["question"]]

	inputs = tokenizer(
		questions,
		context,
		max_length=384,
		truncation=True,
		stride=128,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		padding="max_length",
	)

	sample_map = inputs.pop("overflow_to_sample_mapping")
	example_ids = []

	for i in range(len(inputs["input_ids"])):
		sample_idx = sample_map[i]
		example_ids.append(example["id"][sample_idx])

		sequence_ids = inputs.sequence_ids(i)
		offset = inputs["offset_mapping"][i]
		inputs["offset_mapping"][i] = [
			o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
		]

	inputs["example_id"] = example_ids
	return inputs


def postprocess_qa_predictions(
	examples,
	features,
	all_start_logits,
	all_end_logits,
	tokenizer,
	n_best_size=20,
	max_answer_length=30,
	squad_v2=False
):
	example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
	features_per_example = collections.defaultdict(list)
	for i, feature in enumerate(features):
		features_per_example[example_id_to_index[feature["example_id"]]].append(i)

	predictions = collections.OrderedDict()

	for example_index, example in enumerate(tqdm(examples)):
		feature_indices = features_per_example[example_index]
		min_null_score = None
		valid_answers = []
		context = example["context"]

		for feature_index in feature_indices:
			start_logits = all_start_logits[feature_index]
			end_logits = all_end_logits[feature_index]
			offset_mapping = features[feature_index]["offset_mapping"]

			start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
			end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
			for start_index in start_indexes:
				for end_index in end_indexes:
					if (
						start_index >= len(offset_mapping)
						or end_index >= len(offset_mapping)
						or offset_mapping[start_index] is None
						or offset_mapping[end_index] is None
					):
						continue
					if end_index < start_index or end_index - start_index + 1 > max_answer_length:
						continue

					start_char = offset_mapping[start_index][0]
					end_char = offset_mapping[end_index][1]
					valid_answers.append(
						{
							"score": start_logits[start_index] + end_logits[end_index],
							"text": context[start_char:end_char],
						}
					)
		
		if len(valid_answers) > 0:
			best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
		else:
			best_answer = {"text": "", "score": 0.0}

		predictions[example["id"]] = best_answer["text"]
			
	return predictions


train_dataset = dataset["train"].map(
	preprocess_train,
	batched=True,
	remove_columns=dataset["train"].column_names,
)
train_dataset = train_dataset.shard(num_shards=40, index=0)

validation_dataset = dataset["test"].map(
	preprocess_validation_examples,
	batched=True,
	remove_columns=dataset["test"].column_names,
)
validation_dataset = validation_dataset.shard(num_shards=40, index=0)
validation_dataset.save_to_disk("data")

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
	"qa_model",
	learning_rate=2e-5,
	num_train_epochs=3,
	weight_decay=0.01,
	push_to_hub=False,
	fp16=True,
	#use_cpu=True,
)

squad_metric = evaluate.load("squad")

def compute_metrics(p):

	predicted_char_answers = postprocess_qa_predictions(
		dataset["test"],     # Original examples from the test set
		validation_dataset,  # Processed features (output of .map(preprocess_validation_examples))
		p.predictions,       # Tuple of (start_logits, end_logits)
		tokenizer,
		squad_v2=False
	)

	formatted_references = [
		{"id": ex["id"], "answers": ex["answers"]} for ex in dataset["test"]
	]
	
	# Convert predicted_answers (OrderedDict) to list of dicts for SQuAD metric
	predictions_for_metric = [
		{"id": k, "prediction_text": v} for k, v in predicted_char_answers.items()
	]

	return squad_metric.compute(predictions=predictions_for_metric, references=formatted_references)


trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=validation_dataset,
	tokenizer=tokenizer,
	compute_metrics=compute_metrics,
	)

trainer.train()
trainer.save_model("./qa_model_final")