from google import genai
import os
import glob
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def read_md_files(directory):

	md_files = glob.glob(f"{directory}/*.md")
	documents = []
	for file_path in md_files:
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()
			documents.append({
				"filename":os.path.basename(file_path),
				"content":content
			})

	return (documents)

def generate_qa_pairs(context, num_pairs=3):
	""" Use long context LLM(Gemini-2.0-flash) to generate qa pairs """

	cleaned_context = re.sub(r'[#*`~>\-+_=|]', '', context).strip()
	prompt = f"""
Étant donné le contexte suivant, générez {num_pairs} paires de questions-réponses au format SQUAD.
Chaque réponse doit être extractible du contexte avec le texte exact.
Incluez la position de début de la réponse (index de caractère où la réponse commence dans le contexte).

Contexte : '{cleaned_context}'

Pour chaque paire, suivez exactement ce format :
Context: <contexte>\n
Question: <texte_de_la_question>\n
Answer: <texte_de_la_réponse>\n
Answer_start: <position_entière>

Assurez-vous que les réponses peuvent être trouvées textuellement dans le contexte.
"""

	response = client.models.generate_content(
		model=model,
		contents=prompt
	)

	return (response.text)

def process_res_text(text: str):
	parsed_pairs = []
	lines = text.strip().split('\n')

	curr_pair = {}
	for line in lines:
		line = line.strip()
		if line.startswith("Question:"):
			if curr_pair and "question" in curr_pair and "answer" in curr_pair:
				parsed_pairs.append(curr_pair)
				curr_pair = {}
			curr_pair["question"] = line[len("Question:"):].strip()
		elif line.startswith("Answer:"):
			curr_pair["answer"] = {"text": line[len("Answer:"):].strip()}
		elif line.startswith("Answer_start:"):
			try:
				curr_pair["answer"]["answer_start"] = int(line[len("Answer_start:"):].strip())
			except ValueError:
				curr_pair["answer"]["answer_start"] = 0
	
	if curr_pair and "question" in curr_pair and "answer" in curr_pair:
		parsed_pairs.append(curr_pair)
	
	return parsed_pairs

def main():

	# use langchain to split chunks
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)

	docs_directory = "local_doc"
	documents = read_md_files(docs_directory)
	min_len = 30

	# Generate and save QA pairs for each document
	all_qa_entries = []

	for doc in documents:
		print(f"Processing: {doc['filename']}")
		all_chunks = text_splitter.split_text(doc['content'])

		print(f"Number of chunks for {doc['filename']}: {len(all_chunks)}")

		chunk_index = 0
		for paragraph in all_chunks:
			if (len(paragraph) < min_len): # if the chunk is too small, ignore it 
				continue
			qa_pairs = generate_qa_pairs(paragraph, 1)
			with open("raw_qa_pairs", 'a') as f:
				f.write(qa_pairs + '\n\n----------\n\n')
			qa_pairs = process_res_text(qa_pairs)

			for pair in qa_pairs:
				all_qa_entries.append({
					"context": paragraph,
					"question": pair["question"],
					"answers": {
						"text": [pair["answer"]["text"]],
						"answer_start": [pair["answer"]["answer_start"]]
					},
					"id": f"{doc['filename']}_{chunk_index}_{len(all_qa_entries)}"
				})
			chunk_index += 1

	# Save dataset to a JSON file
	hf_output_file = "corpus/hf_dataset_qa_pairs.json"
	with open(hf_output_file, 'w', encoding='utf-8') as f:
		json.dump({"data": all_qa_entries}, f, indent=2)

	print(f"HuggingFace-ready dataset saved to {hf_output_file}")

if __name__ == "__main__":
	main()

