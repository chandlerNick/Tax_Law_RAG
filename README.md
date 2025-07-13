# RAG with USC 26 - Tax Code RAG System

Basically we take the tax code from the US, parse the legal xml and put each section into a vector database using BERT or a fine tuned BERT.

On inference, we find the sections most relevant to the query and give those as context for the model to answer the query.

To evaluate, we select several prompts (~30-50), some of which have information in the vector DB and some of which are completely off topic. The baseline is the same language model without the RAG context-giving step. Qualitative measures such as "human-interpreted relevance" can be used alongside Quantitative measures such as Precision and Recall for the database information retrieval step.

This project uses Deep Learning when we utilize BERT. Finetuning BERT on a legal dataset is also a modern use of deep learning.

This project is relevant to industry and builds useful skills because companies could have private documents that could be used to augment the generation of a locally hosted (offline) language model, thereby not being "already in the parameters" of the model due to the privacy of the data. They would then be able to leverage the power of a modern LLM while incurring little risk with regards to data exposure.

Finally, two future directions are as follows: 

- If a systematic way to parse the USC lxml is found, this model could be extended to the remainder of the tax code and, if packaged nicely, could be used as a legal aid for lawyers (if it outperforms base chatGPT).

- The tax code idea could be expanded upon by collecting more relevant data pertaining to individual states, counties, cities, and so forth, thus providing a tool for those interested in the tax law of their respective location.

## 🇩🇪 German Tax Law Extension (Granular EStG Focus)

To explore applicability in German-language legal settings, this project was extended to support **EStG (Einkommensteuergesetz)** using a more **granular, single-law approach** rather than the full tax code set. This focused method allows for detailed experimentation with legal language models in a specific domain (individual income tax).

Only **EStG** was used for this prototype stage — laws like UStG, KStG, AO, or ErbStG were excluded to reduce scope and increase control over data granularity and retrieval quality.

### 📘 EStG – Einkommensteuergesetz (Income Tax Act)

Regulates the taxation of income for individuals. It defines types of income, deductions, tax brackets, and how income tax is calculated for residents and those with income in Germany.

## 📚 Overview of Key German Tax Laws

This project could be applied to other core components of the German federal tax system. Below is a summary of the most relevant laws which could be used instead of EStG:

### 📗 UStG – Umsatzsteuergesetz (VAT Act)
Governs value-added tax (VAT) on goods and services. It sets the rules for when VAT is applicable, who must pay it, and the applicable rates (e.g., 19% standard, 7% reduced).

### 📙 KStG – Körperschaftsteuergesetz (Corporate Tax Act)
Covers corporate income tax for legal entities such as GmbHs and AGs. It defines taxable income, exemptions, and a flat corporate tax rate (currently 15%).

### 📒 AO – Abgabenordnung (Fiscal Code)
The general framework for tax procedures in Germany. It applies across all tax types and includes rules on filing, assessment, audits, penalties, deadlines, and appeals.

### 📕 ErbStG – Erbschaftsteuer- und Schenkungsteuergesetz (Inheritance and Gift Tax Act)
Regulates taxation of wealth transfers through inheritance or gifts. Tax rates and exemptions vary depending on the relationship between giver and recipient.

---

These laws are primarily federal and apply uniformly across Germany, with some exceptions (e.g., property tax models at the state level).
