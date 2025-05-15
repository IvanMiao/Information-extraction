import os
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
from pathlib import Path

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def	get_response(pdf_file: Path, client: Mistral):
	"""Télécharge un fichier PDF et obtient la réponse OCR"""
	uploaded_file = client.files.upload(
		file = {
			"file_name": pdf_file.stem,
			"content": pdf_file.read_bytes(),
		},
		purpose="ocr",
	)
	signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

	pdf_response = client.ocr.process(
		document = DocumentURLChunk(document_url=signed_url.url),
		model="mistral-ocr-latest",
		# include_image_base64=True,
	)

	return (pdf_response)

def get_combined_markdown(ocr_response: OCRResponse) -> str:
	"""Combine le markdown de toutes les pages du document"""
	markdowns: list[str] = []
	for page in ocr_response.pages:
		markdowns.append(page.markdown)

	return "\n\n".join(markdowns)

def main():
	"""
	Traiter tous les PDF dans le répertoire local_doc
    et les convertit en fichiers markdown
	"""
	pdf_dir = Path("./local_doc")
	pdf_files = list(pdf_dir.glob("*.pdf"))
	for pdf_file in pdf_files:
		pdf_response = get_response(pdf_file, client)
		content = get_combined_markdown(pdf_response)

		output_path = pdf_file.with_suffix(".md")
		with open(output_path, "w", encoding="utf-8") as f:
			f.write(content)

if __name__ == "__main__":
	main()
