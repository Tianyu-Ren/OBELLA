from model import OBELLA, DotProductAttention, masked_pooling
from tqdm import tqdm
import torch
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
from transformers import QuestionAnsweringPipeline, AutoTokenizer, AlbertForQuestionAnswering
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, Dataset
import os


class Retriever:
    def __init__(self, lucene_index_file):
        self.searcher = LuceneSearcher(lucene_index_file)

    def search_text(self, query: str, k=10):
        hits = self.searcher.search(query, k)

        contents = []

        for i in range(len(hits)):
            doc_id = hits[i].docid
            doc = self.searcher.doc(str(doc_id))
            _ = json.loads(doc.raw())
            contents.append(_["contents"])

        return contents


@torch.inference_mode()
def closed_book_stage(model: OBELLA, test_dataset, device='cuda:0'):
    pred_labels, buffer = [], []
    model = model.to(device)
    for x in tqdm(test_dataset, desc='Closed-Book Stage:'):

        global_idx = 0
        question, references, candidate = x.get("question"), x.get("reference"), x.get("candidate").strip()
        # Multiple-Reference Case. We expect the reference to be a list of strings where each string is a reference
        # answer and is split by '/'. To ensure 24GB GPU memory is enough, we only consider the first 64 references.
        references = references.split('/')[:64]
        questions = [question] * len(references)
        candidates = [candidate] * len(references)

        logits = model(questions, references, candidates)

        y_hat = logits.argmax(dim=-1).cpu().numpy()

        if np.all(y_hat == 2):  # If all y_hats are incorrect
            pred_label = 0
        elif 0 in y_hat:  # If there is at least one y_hat that is correct
            pred_label = 1
        else:
            pred_label = 2
            buffer.append(x)

        pred_labels.append(pred_label)

    return np.asarray(pred_labels), buffer


def open_book_evaluation(model: OBELLA, buffer: list, qa_filter_path=None, knowledge_base_path=None,
                         k=10, theta=0.95, device='cuda:0'):
    if knowledge_base_path is not None:
        retriever = Retriever(knowledge_base_path)
    else:
        # Please follow the instructions to download and index the data and save
        # it as 'factoid_wiki_indices' in the current directory
        retriever = Retriever('factoid_wiki_indices')

    if qa_filter_path is None:
        qa_filter_path = 'ahotrod/albert_xxlargev1_squad2_512'

    qa_model = AlbertForQuestionAnswering.from_pretrained(qa_filter_path)
    tokenizer = AutoTokenizer.from_pretrained(qa_filter_path)
    qa_filter = QuestionAnsweringPipeline(qa_model, tokenizer, device=device)

    predictions = []
    for i in tqdm(buffer, desc='Open-Book Stage:'):
        add_reference = []
        question = i['question']
        candidate = i['candidate']
        context = retriever.search_text(question + ' ' + i['reference'], k)
        answer = qa_filter({'question': [question] * len(context), 'context': context})
        for j_idx, j in enumerate(answer):
            if j['score'] > theta:
                add_reference.append(context[j_idx])
        if len(add_reference) == 0:
            predictions.append(0)
            continue
        questions = [question] * len(add_reference)
        add_reference = [i.split('\n')[1].strip('.') for i in add_reference]
        with torch.no_grad():
            hat = model(questions, add_reference, [candidate] * len(questions))
        pred = hat.argmax(dim=-1)
        predictions.append(1 if 0 in pred else 0)

    return np.asarray(predictions)


def close_then_open_pipeline(model: OBELLA, test_dataset, qa_filter_path=None, knowledge_base_path=None,
                             k=10, theta=0.95, device='cuda:0'):
    pred_labels, buffer = closed_book_stage(model, test_dataset, device)

    open_book_pred = open_book_evaluation(model, buffer, qa_filter_path, knowledge_base_path, k, theta, device)

    pred_labels[pred_labels == 2] = open_book_pred

    accuracy = accuracy_score(test_dataset['label'], pred_labels)
    f1 = f1_score(test_dataset['label'], pred_labels)

    print('Accuracy:', accuracy)
    print('F1:', f1)
    return accuracy, f1, pred_labels


def benchmark_test(obella_checkpoint_path=None, device='cuda:0', benchmark='EVOUNA-NQ', qa_filter_path=None,
                   knowledge_base_path=None, k=10, theta=0.95, save_dir='benchmark_test'):
    assert benchmark in ['EVOUNA-NQ', 'EVOUNA-TQ']
    if obella_checkpoint_path is None:
        obella_checkpoint_path = 'obella_checkpoint/checkpoint'

    model = OBELLA(attention_func=DotProductAttention, pooling=masked_pooling, num_class=3, max_length=256,
                   post_trained_weight=obella_checkpoint_path)

    dataset_names = ['fid', 'gpt35', 'chatgpt', 'gpt4', 'newbing']

    for name in dataset_names:
        print(f'Test on {benchmark}-{name}')
        test_dataset = load_dataset('csv', data_files=f'{benchmark}/{name}.csv', split='train').filter(lambda x: x['candidate'].strip() != '')

        _, __, pred_labels = close_then_open_pipeline(model, test_dataset, device=device, qa_filter_path=qa_filter_path,
                                                      knowledge_base_path=knowledge_base_path, k=k, theta=theta)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        Dataset.from_dict({'pred_labels': pred_labels}).to_json(f'{save_dir}/{benchmark}_{name}_pred_labels.json')

        
benchmark_test()
