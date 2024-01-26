# End-to-End Models for Children’s Speech Recognition (E2ECSR)

This repository provides the recipe for the development of E2E CSR systems for Dutch, German, and Mandarin.

For each language, the repository includes scripts to develop ASR models trained on Adult speech and tested with Child speech.


## Overview: ASR Details
- **Training:** Train on adult Spech and test on adult and child speech
- **Techniques:** Speed Perturbations (SP), Spectral Augmentation (SPaug) and Vocal Tract Length Normalization (VTLN)
- **Architecture:** Conformer-based model without any language model. The ESPnet toolkit and scripts are used for building the ASR model.
- **Evaluation:** Word Error Rate (WER)


## Dutch-CSR
- **cgn_base:** Scripts for training an ASR model with CGN data and testing on CGN and Jasmin datasets.
- **cgn_base_vtln:** Scripts to train ASR model speed perturbations, spectral augmentation and VTLN. VTLN trainng and testing scripts included.
- **Train_VTLN:** Scripts to train a VTLN Model on a given database without training a ASR model

Place the needed scripts in ~/espnet/egs/ to run the ASR models.

**Database:**

|  |  | | |  | #Utterances-#Hours |  | | |
| --- | --- |--- | --- | --- | --- | --- | --- | --- |
| Language | Datasets | Speaking Style | Age Range| #Speakers | Training | Validation | Test-Read | Test-CTS/HMI |
| Dutch  | CGN  | Read-CTS  | 18-65   | 2897  | 704293 (433hrs)  | 70498 (43hrs)  | 409 (0.45hrs)  | 3884 (1.80hrs) \\  
| | Jasmin-DC   | Read-HMI  | 06-13    | 71  | -  | -  | 13104 (6.55hrs)  | 3955 (1.55hrs) |
| | Jasmin-DT   | Read-HMI  | 12-18   | 63  | -  | -  | 9061 (4.90hrs)  | 2723 (0.94hrs) |
| | Jasmin-NnT  | Read-HMI  | 11-18   | 53  | -  | -  | 11545 (6.03hrs)  | 3093 (1.16hrs) |

The repo gives details of the audio files used in training, validation, and testing. The actual database needs to be downloaded from: 
CGN: [CGN Database](https://ivdnt.org/images/stories/producten/documentatie/cgn_website/doc_English/topics/index.htm)
Jasmin: [Jasmin Database](https://www.aclweb.org/anthology/L06-1141/). 

In case of any issue with downloading the database, please contact the developers for access to this corpus.
