This section covers the USC 26 portion of the Vector DB - RAG system

Specifically we look at:
1. Parsing the data
2. Fine tuning BERT
3. Creating the vector DB
4. Evaluating the vector DB
5. Creating the RAG system

The initial parsing was done in initial_parse.ipynb, this was refined and then reused in other files utilizing the data. We essentially parse at the section level, taking all text at this level as content and adding the metadata tags.

The fine tuning of BERT was done in the FineTuneBERT directory. All that must change is the pathing in the python script to change where the data is and where the results are stored if one wants to recreate our work. Our results are stored in the usc26/FineTuneBERT_USC26/FTBERT_HPO_Results. The fine tuned model is on HuggingFace as an embedding model: 


