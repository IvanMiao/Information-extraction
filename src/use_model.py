from transformers import pipeline

model_checkpoint = "qa_model_final"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
Roman XIXeXXe siècle

 Fréq.  Forme  Gastronomie  Classe des <br mets festifs 
 ::  ::  ::  :: 
 101  caviar  /luxueux/  caviar 
 15  champagne  /estimé/  champagne 
 10  foie gras  /fin/  foie gras 
 9  repas    truffe 
 8  petit    crustacés 
 6  grains    saumon fumé 
 6  noir    ... 
 5  bon     
 5  gris     
 5  Iran     
 5  poulet     
 5  vodka     
 4  beau     
 4  gelée     
 4  manger     
 4  maître     
 4  soir     
 4  tartines     
 4  truffes  /recherché/   

 Sèmes et classes sémantiques Catégorisation paradigmatique 

Roman XIXeXXe siècle
Fréq. Forme
 """
question = "Quelle est la fréquence du mot caviar ?"
print(question_answerer(question=question, context=context))