from transformers import pipeline

model_checkpoint = "qa_model_final"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
Votre contexte
 """
question = "[Votre question]"
print(question_answerer(question=question, context=context))
