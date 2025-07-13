# 🇩🇪 German Tax Law RAG – EStG Focus

This module is a more granular German-language version of the main RAG-based legal retrieval system.  
It focuses exclusively on the **Einkommensteuergesetz (EStG)** — Germany’s Income Tax Act — as a testbed for legal document retrieval and generation in a controlled, single-law setting.

Also for this part of the project, two different learning rates for BERT and the added classifier layer are used to try out the effects of keeping the pretrained model more similar to how it was before.

---

## 📌 Project Scope

To explore applicability in German-language legal settings, this project was extended to support **EStG (Einkommensteuergesetz)** using a more **granular, single-law approach** rather than including the entire German tax code.

Only **EStG** was used for this prototype — laws such as UStG, KStG, AO, or ErbStG were **not** included.  
This focused setup improves the precision and clarity of experiments around vector retrieval, prompt quality, and LLM-based legal answering.

### 📘 EStG – Einkommensteuergesetz (Income Tax Act)

The EStG regulates the taxation of income for individuals.  
It defines:
- Types of income
- Allowable deductions
- Tax brackets
- Calculation methods for residents and foreign income earners

The xml file can be downloaded [here](https://www.gesetze-im-internet.de/estg/).

---

## Project Steps

### 1. Clone the repository, create venv and install requirements
```bash
git clone https://github.com/chandlerNick/Tax_Law_RAG.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd Tax_Law_RAG/german_tax_law
```

### 2. Fine Tune BERT
- Adjust data paths for reading and output
- In folder hpo-finetuning run ./runHPO.sh (adjust hyperparameters as wished)
- Examine FTBERT_HPO_Results for best hyperparameters in the sweep.
- Go through the file EvalPublishFTBERT_EStG/evaluate_FT_BERT.ipynb, change the hyperparameters to the best ones found.
- Record the metric from the fine tuned BERT and non-fine tuned BERT
- Go through the file EvalPublishFTBERT_EStG/publish_FT_BERT.ipynb and publish the model to hugging face

### 3. Vector DB/RAG
- Go through the VectorDB notebook vector_db/VectorDB.ipynb, only change the data and model. Record retrieval results.
- Go through the RAG notebook vector_db/RAG.ipynb, only change the data and model. Record results.

### 4. For Reference
Our HPO results are in the folder hpo-finetuning/HPO_results.
The finetuned models can be found on Huggingface:
- [FT-BERT-EStG](https://huggingface.co/ninoid/EStG_Subtitle_Classification_BERT)
- [FT-BERT-EStG Sentence Trandformers](https://huggingface.co/ninoid/sentence-transformers-EStG-bert)


