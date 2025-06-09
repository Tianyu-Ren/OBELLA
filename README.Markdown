# OBELLA



## OBELLA for OBELLAM Training



### **Data Access**

Our OBELLA is available for download [here](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40414335_ads_qub_ac_uk/EqZxN0F8UERGmjijE5NaNUQBo2T_uaE7pCtRkWGBVdHQQg?e=B8Fq60)

### Format

All dataset files are in JSON or JSONL format. Below is an example entry:

```json
{
  
 "question": "What aspects of language does linguistics study beyond form and meaning?",
 "reference":"Form and meaning are the only aspects of language linguistics is concerned with.",
 "candidate":"Linguistics is the scientific study of language, and involves an analysis of language form, 								language meaning, and language in context. The earliest activities in the documentation and 								description of language have been attributed to the 4th century BCE Indian grammarian 							 				P\u0101\u1e47ini, who wrote a formal description of the Sanskrit language in his 														\"A\u1e63\u1e6d\u0101dhy\u0101y\u012b \".",
 "label":2

}

```

**Note** that label of each example adheres to the mapping: `{0: 'Correct', 1: Neurtal, 2: 'Incorrect'}`

### **Training OBELLAM on OBELLA**

#### Version Information:

```
datasets==2.21.0
torch==2.1.2+cu121
tqdm==4.66.5
transformers==4.44.0
```

#### Struture

```
OBELLA-TRAIN/
│
├── model.py         
├── load_data.py      
├── train.py       
├── utils.py    
├── OBEDATA/     # Download it using the provided link and save it to OBELLA-TRAIN
│   └── OBEDATA-S.jsonl      
│   └── OBEDAtA-L-TRAIN.json
│   └── OBEDATA-L-DEV.json
└── run.sh  # This provides the demo for training OBELLA on OBEDATA-L using our paper's settings
```

Simply run `source run.sh` to train a supervised ODQA evaluation metric using **OBEDATA-L**.

## Using OBELLAM for Open-Domain Question Answering Evaluation

Considering the high training cost, we provide a pre-trained OBELLAM checkpoint. You can test OBELLA’s performance on the EVOUNA benchmark using the provided evaluation script.

### **Download Links**

**OBELLAM Checkpoint and Factoid Wiki Indices:** [link](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40414335_ads_qub_ac_uk/EnIjcWAWDD1MgxUBeSRK5tkBsbIGEr16bZrapvQ1ZaqlCg?e=CcV6Sp)

### **Running the Evaluation**

Execute the following command to evaluate OBELLAM on the EVOUNA benchmark:

```bash
python OBELLA-EVAL/evaluation.py
```



### Version Information

```
datasets==2.21.0
numpy==2.1.3
pyserini==0.43.0
scikit_learn==1.5.2
torch==2.1.2+cu121
tqdm==4.66.5
transformers==4.44.0
```

#### **Evaluation Directory Structure**

```
OBELLA-EVAL/
│
├── model.py         
├── evaluation.py            
├── factoid_wiki_indices    # Download and place here
├── obella_checkpoint
│   └── checkpoint          # Place downloaded OBELLA checkpoint here
├── EVOUNA-NQ/             
│   ├── fid.csv      
│   ├── chatgpt.csv
│   ├── gpt35.csv
│   ├── gpt4.csv
│   └── newbing.csv
└── EVOUNA-TQ/             
    ├── fid.csv      
    ├── chatgpt.csv
    ├── gpt35.csv
    ├── gpt4.csv
    └── newbing.csv
```

