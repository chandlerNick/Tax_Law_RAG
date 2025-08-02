# Project overview

In this project, we built RAG systems using BERT and Qwen to answer questions about tax law. We compared two verctor databases (Annoy and FAISS) as well as fine tuned the embedding model, BERT on a document classification task to compare performance to the base BERT. This was a part of a Deep Learning class at the BHT Berlin and required the use of deep learning techniques in the fine tuning process of our model. It also developed skills in Natural Lanugage Processing (NLP) as we had relatively complex text data and worked with two language models. Additionally, it exercised skills in systems thinking (the RAG architecture) and use of GPU compute infrastructure (kubernetes and docker). While the project is "scientific" in nature, it still demonstrates the ability to implement a RAG system for a specific use case, and fine tune BERT. For a detailed overview of the project (and a bit of teaching since it extended the scope of our class) see `Term_Project.pdf`.


## USC 26 American Tax Law

This law is used to govern the American tax system and is a part of the broader United States Code (USC). It comes in lxml format.

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
