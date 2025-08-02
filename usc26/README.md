This section covers the USC 26 portion of the Vector DB - RAG system

Specifically we look at:
1. Parsing the data
2. Fine tuning BERT
3. Creating the vector DB
4. Evaluating the vector DB
5. Creating the RAG system

The initial parsing was done in initial_parse.ipynb, this was refined and then reused in other files utilizing the data. We essentially parse at the section level, taking all text at this level as content and adding the metadata tags.

The fine tuning of BERT was done in the FineTuneBERT directory. All that must change is the pathing in the python script to change where the data is and where the results are stored if one wants to recreate our work. Our results are stored in the usc26/FineTuneBERT_USC26/FTBERT_HPO_Results. The fine tuned model is on HuggingFace as an embedding model: 


---

Basic Process:

- Fine Tune BERT
1. Ensure dataset pathing is correct
2. In FineTuneBERT_USC26 run `./runHPO.sh`
3. Examine FTBERT_HPO_Results for best hyperparameters in the sweep.
4. Go through the evaluate and fine tune BERT file, change the hyperparameters to the best ones found.
5. Record the metric from the fine tuned BERT and non-fine tuned BERT
6. Go through the publish FT BERT file and publish the model to hugging face

- Vector DB/RAG
1. Go through the VectorDB notebook, only change the data and model. Record retrieval results.
2. Go through the RAG notebook, only change the data and model. Record results.


Note, this code is not production grade, rather research grade. That is, it is intended to get the results as quickly as possible for records.
