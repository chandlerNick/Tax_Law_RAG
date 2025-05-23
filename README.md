# RAG with USC 26 - Tax Code RAG System

Basically we take the tax code from the US, parse the legal xml and put each section into a vector database using BERT or a fine tuned BERT.

On inference, we find the sections most relevant to the query and give those as context for the model to answer the query.

To evaluate, we select several prompts (~30-50), some of which have information in the vector DB and some of which are completely off topic. The baseline is the same language model without the RAG context-giving step. Qualitative measures such as "human-interpreted relevance" can be used alongside Quantitative measures such as Precision and Recall for the database information retrieval step.

This project uses Deep Learning when we utilize BERT. Finetuning BERT on a legal dataset is also a modern use of deep learning.

This project is relevant to industry and builds useful skills because companies could have private documents that could be used to augment the generation of a locally hosted (offline) language model, thereby not being "already in the parameters" of the model due to the privacy of the data. They would then be able to leverage the power of a modern LLM while incurring little risk with regards to data exposure.

Finally, two future directions are as follows: 

- If a systematic way to parse the USC lxml is found, this model could be extended to the remainder of the tax code and, if packaged nicely, could be used as a legal aid for lawyers (if it outperforms base chatGPT).

- The tax code idea could be expanded upon by collecting more relevant data pertaining to individual states, counties, cities, and so forth, thus providing a tool for those interested in the tax law of their respective location.

