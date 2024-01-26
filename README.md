# E2ECSR
End-to-End Models for Children’s Speech Recognition

This repository provides the recipe for development of E2E CSR system for Dutch, German and Mandarin.

For each language, the scripts to develop and ASR when trained on Adult speech and tested with Child speech.

**ASR Models:**


├── Dutch-CSR

│   ├── cgn_base

│   ├── cgn_base_vtln

│   └── Train_VTLN


**Database:**

|  |  | | |  | #Utterances-#Hours |  | | |
| --- | --- |--- | --- | --- | --- | --- | --- | --- |
| Language | Datasets | Speaking Style | Age Range| #Speakers | Training | Validation | Test-Read | Test-CTS/HMI |
| Dutch  | CGN  | Read-CTS  | 18-65   | 2897  | 704293 (433hrs)  | 70498 (43hrs)  | 409 (0.45hrs)  | 3884 (1.80hrs) \\  
| | Jasmin-DC   | Read-HMI  | 06-13    | 71  | -  | -  | 13104 (6.55hrs)  | 3955 (1.55hrs) |
| | Jasmin-DT   | Read-HMI  | 12-18   | 63  | -  | -  | 9061 (4.90hrs)  | 2723 (0.94hrs) |
| | Jasmin-NnT  | Read-HMI  | 11-18   | 53  | -  | -  | 11545 (6.03hrs)  | 3093 (1.16hrs) |


