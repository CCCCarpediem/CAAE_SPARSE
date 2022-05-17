# CAAE-Sparse
A brand new multi-label classifier based on deep learning. Get superior results on medical data, solve ICU patients' prescription problems. 

#code file
'C2AE' : main structure of the model.
'MIMIC' :process the MIMIC-III data[1], including HDMI based reconstruction one-hot encoding, word embedding etc.
'test': experiments based on other multi-label classification models.

#prerequisites
Python                             3.7.7
numpy                              1.16.2
tensorflow                         1.14.0
scikit-learn                       0.20.3
pandas                             0.24.2


[1] MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L,
Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016).
DOI: 10.1038/sdata.2016.35.
