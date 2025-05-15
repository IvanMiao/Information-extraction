# Rapport

## Introduction

Ce projet vise à développer un système de questions-réponses (QA) pour la langue française. Une part de ce travail a consisté à expérimenter la génération automatique de paires question-réponse. Pour ce faire, on a mis en œuvre un pipeline complet : conversion de documents PDF au format Markdown par reconnaissance optique de caractères (OCR), segmentation (chunking) des textes obtenus, puis utilisation d'un grand modèle de langage (LLM) pour générer les paires QA, qui ont ensuite été structurées en un corpus compatible avec les datasets Hugging Face. En parallèle, un modèle de QA pré-entraîné a été fine-tuné. L'évaluation de ce dernier a été effectuée en comparant ses scores F1 sur deux ensembles de données : d'une part, la partition de validation du jeu de données public utilisé pour l'entraînement, et d'autre part, un corpus de test généré par le LLM à partir de nos documents locaux.

## État de l’art

### Article 1: SQUAD: 100,000+ Questions for Machine Comprehension of Text

#### Contexte de l’article

L'article introduit le Stanford Question Answering Dataset (SQuAD), un ensemble de données destiné à la compréhension de lecture par des machines (Reading Comprehension, RC). Les auteurs soulignent qu'au moment de la publication, les ensembles de données existants pour la RC présentaient des limitations : soit ils étaient de haute qualité mais trop petits pour entraîner des modèles gourmands en données, soit ils étaient volumineux mais générés de manière semi-synthétique et ne reflétaient pas la complexité des questions réelles de compréhension de texte. SQuAD a été créé pour combler ce manque en proposant un dataset à la fois vaste et de haute qualité, où les réponses aux questions sont des segments de texte (spans) extraits directement des passages de référence. L'objectif était de fournir une ressource robuste pour entraîner et évaluer des modèles de RC, et de stimuler la recherche dans ce domaine.

#### Expérimentations

Les auteurs ont sélectionné 536 articles de Wikipédia, couvrant un large éventail de sujets. Pour garantir la qualité, ils ont utilisé le PageRank interne de Wikipédia pour obtenir les 10 000 articles les plus importants, puis en ont échantillonné aléatoirement 536. Les paragraphes de moins de 500 caractères, ainsi que les images, figures et tableaux ont été écartés.

Des travailleurs humains (via la plateforme Daemo et Amazon Mechanical Turk) ont été chargés de poser jusqu'à 5 questions par paragraphe et d'identifier la réponse correspondante en surlignant un segment de texte dans ce même paragraphe. Des instructions claires ont été données pour encourager la formulation de questions dans leurs propres mots et éviter la simple copie de phrases.

Pour évaluer la performance humaine et rendre l'évaluation plus robuste, au moins deux réponses supplémentaires ont été collectées pour chaque question des ensembles de développement et de test.

Les auteurs ont analysé SQuAD pour comprendre la diversité des types de réponses (dates, entités nommées, phrases nominales, etc.) , les types de raisonnement nécessaires pour répondre aux questions (variation lexicale, variation syntaxique, raisonnement multi-phrases) , et la divergence syntaxique entre les questions et les phrases contenant les réponses.

#### Résultats

Les principaux résultats présentés dans l'article sont :

- Performance du modèle de régression logistique : Le modèle de régression logistique a atteint un score F1 de 51.0% sur l'ensemble de test, ce qui représentait une amélioration significative par rapport à une baseline simple (environ 20%). 

- Performance humaine : La performance humaine sur la même tâche a été évaluée à un score F1 de 86.8%. L'écart important entre la performance du modèle et celle des humains indiquait que SQuAD constituait un défi substantiel pour les systèmes de RC.

#### Pourquoi cet article est pertinent dans le cadre de votre projet

Mon projet vise à générer des paires de questions-réponses (QA) de type SQuAD. Cet article est la référence qui définit SQuAD, sa structure (question, contexte, réponse comme segment de texte), et la méthodologie de sa création. Comprendre cet article est fondamental pour savoir quel type de données je cherche à générer.

De plus, SQuAD a établi des métriques d'évaluation standards pour ce type de tâche de QA extractive, notamment l'Exact Match (EM) et le score F1.

Sur Hugging Face, SQuAD est l'un des datasets de QA les plus populaires et est nativement supporté par la bibliothèque `datasets` de Hugging Face. Comprendre sa structure m'aidera à formater correctement mes propres données générées pour les charger en tant que Dataset Hugging Face.


### Arcitle 2: Prompting-based Synthetic Data Generation for Few-Shot Question Answering

#### Contexte de l’article

Cet article aborde la problématique de la rareté des données pour l'entraînement de modèles de questions-réponses par lecture de machine (MRQA), particulièrement dans des contextes à faible nombre d'exemples (few-shot) et pour des domaines spécifiques. Les auteurs soutiennent que l'annotation de données, coûteuse et chronophage, peut être atténuée en exploitant les connaissances générales des grands modèles de langage (LLMs) pour générer des données synthétiques de haute qualité. L'étude se concentre sur la QA extractive, où la réponse est un segment de texte extrait du contexte.

#### Expérimentations

L'approche proposée repose sur une génération de données synthétiques via le "prompting" de LLMs. D'abord, des candidats réponses sont échantillonnés à partir de documents non étiquetés en utilisant la reconnaissance d'entités nommées (NER), une méthode simple et indépendante du domaine. Ensuite, un LLM pré-entraîné (T5-v1.1 large) est sollicité avec une invite (prompt) contenant le contexte et la réponse échantillonnée pour générer une question correspondante. Les paires QA générées sont ensuite filtrées par des règles et par un critère de cohérence, où un modèle MRQA vérifie si la question générée permet de retrouver la réponse initiale. Finalement, un modèle MRQA (T5-v1.1 large avec pré-entraînement RSS) est entraîné d'abord sur ces données synthétiques, puis affiné avec les quelques données étiquetées disponibles. Les expériences ont été menées sur SQuAD et d'autres jeux de données MRQA en configuration few-shot, évaluant les performances avec le score F1. Une étude utilisateur a également été menée pour évaluer la qualité des paires QA générées.

#### Résultats

L'approche de génération de données synthétiques par prompting a surpassé de manière constante les méthodes de pointe antérieures en QA few-shot sur divers jeux de données et pour différentes tailles d'échantillons, démontrant notamment de fortes capacités en zero-shot. L'utilisation de réponses échantillonnées par NER s'est avérée presque aussi performante que l'utilisation de réponses de référence (gold answers) pour la génération de questions. La méthode a également montré une bonne capacité de généralisation à différents domaines. L'étude utilisateur a révélé qu'avec 128 échantillons étiquetés pour entraîner le modèle de génération, la qualité des données synthétiques était comparable à celle des données annotées par des humains pour le dataset NewsQA.

#### Pourquoi cet article est pertinent dans le cadre de votre projet

Cet article est directement lié à mon projet car il propose une méthode détaillée pour utiliser les LLMs (spécifiquement T5 via prompting) afin de générer des paires QA synthétiques, ce qui est une étape clé de ma démarche. La méthodologie en deux étapes (échantillonnage de réponses puis génération de questions conditionnée au contexte et à la réponse) offre une approche concrète que je peux adapter, notamment en utilisant le contenu MD extrait par OCR comme source de contextes et de réponses potentielles. Les techniques de filtrage (par règles et par cohérence) sont également des éléments pratiques pour améliorer la qualité des paires QA que l'LLM générera. Bien que je prévoie d'utiliser DistilBERT pour l'entraînement final, le succès de T5 dans la phase de génération de données est encourageant.


### Article 3: Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs

#### Contexte de l’article

Cet article s'attaque au problème de la pénurie de données étiquetées pour l'entraînement de modèles de questions-réponses (QA), une tâche coûteuse en temps et en efforts humains. Il se concentre sur la génération automatique de paires QA diversifiées et cohérentes à partir de contextes textuels non structurés. Les auteurs notent que les approches existantes, souvent basées sur des modèles séquence-à-séquence, tendent à produire des paires QA génériques et peu variées, ce qui est sous-optimal étant donné la richesse d'information souvent présente dans un seul contexte. Un autre défi majeur est d'assurer la cohérence sémantique entre la question générée et sa réponse.

#### Expérimentations

Les auteurs proposent un modèle dénommé Info-HCVAE (Information-Maximizing Hierarchical Conditional Variational Autoencoder). Ce modèle repose sur un autoencodeur variationnel conditionnel hiérarchique (HCVAE) avec deux espaces latents distincts (Gaussien pour la question, catégoriel pour la réponse) pour favoriser la diversité, où l'espace latent de la réponse est conditionné par celui de la question. Le processus génère d'abord une réponse à partir du contexte, puis une question à partir du contexte et de la réponse. Pour garantir la cohérence, un régularisateur InfoMax est introduit, visant à maximiser l'information mutuelle entre la question et la réponse générées, estimée via une approximation neuronale basée sur la divergence de Jensen-Shannon. Le modèle intègre des Bi-LSTMs, des MLPs et des embeddings BERT. L'évaluation est réalisée à l'aide de métriques spécifiques : la QA-based Evaluation (QAE), où un modèle QA (BERT-base) est entraîné sur les données synthétiques et testé sur des données humaines, et la Reverse QAE (R-QAE), où un modèle QA entraîné sur des données humaines est testé sur les données synthétiques, une faible R-QAE (avec une QAE élevée) indiquant des paires générées nouvelles et diverses. Les expériences ont porté sur les datasets SQuAD, Natural Questions (NQ), et TriviaQA.

#### Résultats

L'Info-HCVAE a significativement surpassé les modèles de génération de QA de référence en termes de QAE sur SQuAD, NQ et TriviaQA, tout en obtenant des scores R-QAE plus bas, ce qui indique la génération de paires QA à la fois de haute qualité et diversifiées. Le régularisateur InfoMax s'est avéré efficace pour améliorer la cohérence des paires QA et la performance globale. L'Info-HCVAE a atteint des performances élevées en utilisant un volume de paires QA nettement inférieur à celui des baselines. Une évaluation humaine a confirmé la supériorité des paires QA générées par Info-HCVAE en termes de diversité, de cohérence et de qualité globale. Le modèle a également démontré sa capacité à générer plusieurs questions distinctes et pertinentes pour une même réponse et un même contexte (génération un-à-plusieurs). En apprentissage semi-supervisé, l'utilisation des données générées par Info-HCVAE a amélioré les performances d'un modèle QA BERT-base sur les trois datasets.

#### Pourquoi cet article est pertinent dans le cadre de votre projet

Cet article est très pertinent car il met l'accent sur la génération de paires QA diverses à partir d'un unique contexte, un aspect important pour entraîner des modèles QA robustes ; la capacité de l'Info-HCVAE à effectuer une génération "un-à-plusieurs" (one-to-many QG) est une idée clé. La question de la cohérence entre questions et réponses est également centrale, et bien que les LLMs modernes puissent offrir une meilleure cohérence intrinsèque, les concepts de maximisation de l'information mutuelle ou les techniques de filtrage/raffinement peuvent être bénéfiques pour mes données générées. L'approche probabiliste du VAE pour introduire de la variété, bien que distincte du prompting direct d'un LLM, souligne l'importance de viser la diversité, ce qui pourrait me inciter à explorer des stratégies pour que l'LLM produise des questions variées pour un même chunk de texte. Les métriques d'évaluation QAE et R-QAE offrent des moyens sophistiqués d'évaluer la qualité et la nouveauté de mon propre jeu de données. Enfin, la technique de raffinement des réponses générées à l'aide d'un modèle QA existant est une méthode pratique que je peux adapter.


## Données/Corpus

L'intention initiale était d'utiliser principalement des paires de questions-réponses (QA) auto-générées. Cependant, la quantité de données pouvant être produites à partir de mes documents locaux s'est avérée insuffisante pour un entraînement robuste. Par conséquent, j'ai opté pour le jeu de données `CATIE-AQ/frenchQA` disponible sur Hugging Face comme corpus d'entraînement principal. Étant donné la taille considérable de ce dataset, qui aurait nécessité plusieurs centaines d'heures de calcul pour l'entraînement, j'ai décidé de le partitionner pour n'utiliser qu'une fraction de 1/40 de sa taille originale pour l'entraînement.

Concernant l'évaluation, la partie "validation" du dataset original `CATIE-AQ/frenchQA` a également été réduite à 1/40 (5016/200617) de sa taille pour constituer un premier ensemble de test. Parallèlement, les paires QA que j'ai moi-même générées constituent un second ensemble de validation. Ce dernier a pour objectif de vérifier si le modèle entraîné conserve une précision généralisable face à des contextes et des questions de nature potentiellement très différente. Ainsi, je dispose de deux ensembles de test distincts : l'un issu du dataset original et l'autre produit par mon propre pipeline `(PDF -> OCR -> Markdown -> Chunking, puis LLM -> Paires QA (contexte, question, réponse))`.

## Méthodes et expérimentations

### génération de données

Pour la génération de données, j'ai eu recours aux API gratuites de Mistral et Gemini. J'ai utilisé Mistral OCR pour convertir plusieurs fichiers PDF locaux en français au format Markdown, avec une précision que j'ai jugée satisfaisante. Par la suite, j'ai employé le framework Langchain pour segmenter (chunk) les textes Markdown obtenus. Chaque segment a ensuite été transmis à `gemini-2.0-flash` avec un prompt détaillé afin qu'il génère des paires QA en français dans le format requis. Le contenu généré a été extrait sous forme de dictionnaires Python, puis sauvegardé en fichiers JSON pour faciliter la construction d'un `Dataset` Hugging Face. Initialement, j'avais envisagé de tirer parti de la capacité de Gemini à traiter de longs contextes pour générer directement plusieurs paires QA à partir d'un document entier. Cependant, par souci de précision et pour limiter les risques d'hallucinations du modèle, j'ai préféré opter pour une approche consistant à soumettre de petits segments de texte (chunks) avec un prompt identique à chaque fois, afin de mieux contrôler la conformité des réponses aux instructions.

### entraînement du modèle

Concernant l'entraînement du modèle, je me suis appuyé sur la documentation de la bibliothèque Transformers de Hugging Face relative au question answering, ainsi que sur les contenus du cours LLM de Hugging Face dédiés à cette tâche. J'ai utilisé la bibliothèque Transformers et choisi `distilbert/distilbert-base-multilingual-cased` comme modèle pré-entraîné à fine-tuner. Ce choix a été motivé par plusieurs facteurs :

1. Il s'agit d'une version allégée de BERT, offrant des performances comparables avec une perte de capacité limitée, ce qui permet un entraînement plus rapide sur des ressources matérielles restreintes (mon ordinateur personnel) en utilisant le GPU.

2. C'est un modèle multilingue, donc plus adapté à l'entraînement sur du contenu non anglophone.

3. Il s'agit d'un modèle "base", c'est-à-dire non préalablement fine-tuné sur des datasets comme `SQuAD`, ce qui permet de mieux évaluer l'efficacité du fine-tuning réalisé.

Dans un premier temps, j'ai tenté d'entraîner le modèle avec mon propre jeu de données auto-généré (environ 200 entrées). Le modèle résultant s'est avéré quasiment incapable d'accomplir la tâche de QA. J'ai ensuite basculé vers l'utilisation de la fraction (1/40) du dataset `CATIE-AQ/frenchQA`(5016 entrées). Après un peu plus de deux heures d'entraînement, le modèle obtenu a montré des performances remarquablement bonnes. Sur mon propre ensemble de validation, les réponses fournies par le modèle étaient proches, voire identiques, aux réponses de référence.


## Evaluation et résultats

**Script pour l'evaluation: `evaluate.py`**

| Validation Set                   | F1 Score |
| -------------------------------- | -------- |
| CATIE-AQ/frenchQA [validation]   | 0.4152   |
| Ensemble de test généré par LLM  | 0.3769   |

## Perspectives d’amélioration

Avec davantage de ressources, tant budgétaires que matérielles, plusieurs améliorations pourraient être envisagées pour perfectionner l'ensemble du flux de travail. Il serait possible d'exploiter des jeux de données plus volumineux, d'utiliser des modèles pré-entraînés de plus grande taille, ainsi que des API de LLM et des technologies OCR plus performantes. Cela permettrait, par exemple, de générer rapidement des paires question-réponse à partir d'un grand nombre de documents locaux, constituant ainsi un corpus d'entraînement comptant potentiellement des dizaines de milliers d'entrées. De plus, des ressources faciliteraient la comparaison des performances de différents modèles.

Avec plus de temps et une expertise approfondie, il serait pertinent d'explorer plus en détail les stratégies de segmentation (chunking) proposées par divers frameworks. Une autre piste consisterait à développer un algorithme personnalisé pour nettoyer plus efficacement les textes Markdown complexes issus de l'OCR, afin d'optimiser la qualité des données en entrée du LLM et du modèle de QA.


## Difficultés rencontrées pendant le projet

1. Nettoyage et formatage des données issues de l'OCR

La conversion des documents PDF en Markdown via OCR (Mistral OCR) a introduit des artefacts et des incohérences de formatage. Bien que le Markdown soit structuré, un effort de nettoyage a été nécessaire pour garantir que le texte transmis au LLM pour la génération des QA et utilisé comme contexte soit clair et pertinent.

Cependant, il reste difficile de nettoyer entièrement le texte. D’un côté, on utilise le format Markdown pour structurer davantage le contenu, afin d’aider le LLM à mieux le comprendre et à générer des paires de questions-réponses plus adaptées — par exemple, en insérant des tableaux. Mais d’un autre côté, le Markdown introduit également des symboles superflus. Il est donc difficile de trouver un bon équilibre entre texte brut et Markdown.

2. Des réponses générées par le LLM

L'une des difficultés majeures a été de s'assurer que les paires question-réponse générées par `gemini-2.0-flash` étaient non seulement grammaticalement correctes, mais aussi que la réponse était factuellement soutenue par le segment de contexte fourni et que la question était pertinente pour cette réponse. Il y aurait des risques d'hallucination, même avec des prompts détaillés et des contextes courts.

3. Définition d'une stratégie de segmentation (chunking) optimale

Le choix de la taille et de la méthode de segmentation des documents Markdown a été un défi. Des segments trop petits risquaient de priver le LLM du contexte nécessaire pour générer des questions et réponses pertinentes et complètes. Inversement, des segments trop longs pouvaient introduire du bruit.

J’ai d’abord tenté d’écrire manuellement une fonction de chunking, mais la logique était difficile à généraliser et les résultats peu satisfaisants. Je me suis alors tourné vers les `text splitters` proposés par le framework `LangChain`. Bien qu’il en existe un spécifiquement conçu pour le format Markdown, le fait que mes fichiers Markdown proviennent d’un OCR automatique compliquait les choses : la hiérarchie des titres y est parfois incohérente, ce qui rendait cette approche moins fiable.

J’ai donc opté pour le `RecursiveCharacterTextSplitter`, plus robuste face à une structuration incertaine. Toutefois, je n’ai pas encore trouvé de stratégie de segmentation qui garantisse à la fois une bonne cohérence sémantique et une longueur de segment adaptée aux textes.

4. Temps d'entraînement

L'entraînement des modèles de type Transformer, même des versions distillées comme `distilbert-base-multilingual-cased`, rdemeure exigeant en ressources de calcul. Malgré l'utilisation d'un GPU et la réduction significative de la taille du dataset `CATIE-AQ/frenchQA` (à 1/40), le temps d'entraînement sur un ordinateur personnel équipé d’une carte graphique GTX1650 est resté une contrainte importante (plus de deux heures). 

Cela a limité la capacité à expérimenter avec différents hyperparamètres, une plus grande portion du dataset, ou des modèles plus grands, ce qui aurait pu potentiellement améliorer les performances. 

La génération de données avec les API LLM, bien que gérable pour la quantité produite, aurait également pu devenir un goulot d'étranglement en cas de montée en volume.


## Conclusion

En conclusion, ce projet a exploré la chaîne complète de développement d'un système de questions-réponses en français, de la création de données synthétiques à l'ajustement fin et l'évaluation d'un modèle. Le pipeline de génération de données (OCR -> chunking -> LLM -> corpus structuré) a permis de produire un ensemble de paires QA à partir de documents locaux. Le modèle `distilbert-base-multilingual-cased`, après fine-tuning sur une fraction du dataset `CATIE-AQ/frenchQA`, a atteint un score F1 de 0.4152 sur la partition de validation de ce dernier, et de 0.3769 sur notre corpus généré. Ces résultats, obtenus malgré les défis liés à la qualité des données OCR et aux contraintes de ressources, soulignent le potentiel de cette approche. Les difficultés rencontrées et les pistes d'amélioration identifiées ouvrent la voie à de futurs travaux pour affiner la qualité des données générées et optimiser les performances des modèles de QA en français.
