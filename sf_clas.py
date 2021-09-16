#!/usr/bin/env python
# coding: utf-8

# In[5]:


from collections import defaultdict
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import torch


# In[6]:


sklearn.__version__


# In[7]:


import pandas as pd


# In[8]:


import pickle
from sklearn.metrics import f1_score


# In[9]:


import spacy
nlp = spacy.load("en_core_web_sm")


# In[10]:


with open('labeled data.json') as data:
    file = json.load(data)


# In[11]:


import re


# In[12]:


import nltk 
from nltk import word_tokenize


# In[13]:


dialogues = [] 
for d in file[:2]:
    samples = defaultdict(dict)
    result = d['completions'][0]['result']
    texts_without_labels = d['data']['text']
    for sample in result:
        speaker = texts_without_labels[int(sample['value']['start'])]['speaker']
        samples[sample['id']]['speaker'] = speaker
        samples[sample['id']]['text'] = sample['value']['text']
        samples[sample['id']]['start'] = int(sample['value']['start'])
        if 'paragraphlabels' in sample['value']:
            samples[sample['id']]['paragraphlabels'] = sample['value']['paragraphlabels'][0]
        if 'choices' in sample['value']:
            samples[sample['id']]['choices'] = sample['value']['choices'][0]
    
    sorted_samples = sorted([(samples[sample_id]['start'], sample_id) for sample_id in samples])
    texts = []
    labels = []
    speakers = []
    for _, sample_id in sorted_samples:
        if samples[sample_id]['text'] != 'PAUSE':
            texts.append(str(samples[sample_id]['text']).replace('\n', ''))
            speakers.append(samples[sample_id]['speaker'])
            paragraph_labels = samples[sample_id].get('paragraphlabels', '')
            choices = samples[sample_id].get('choices', '')
            labels.append(paragraph_labels + '.' + choices)
    dialogues.append((texts, labels, speakers))


# In[14]:


train_data = dialogues[1][0]
test_data = dialogues[0][0]


# In[15]:


train_labels = dialogues[1][1]
test_labels = dialogues[0][1]


# In[16]:


def delete_odds(list_with_lines):
    for i in range(len(list_with_lines)):        
        if 'them' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('them', ' them ')
        if ' em ' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace(' em ', ' them ')
        if 'laugh' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('laugh', '')
        if 'uh?' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('uh?', '')
        if 'ʔuh' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('ʔuh', '')
        if 'ʔ' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('ʔ', '')
    return list_with_lines


# In[17]:


previous_lines_sust=[]
sustains=[]
sus_tags=[]
for i in range(len(train_data)):
    if 'Sustain' in train_labels[i]:
            previous_lines_sust.append(train_data[i-1])
            sustains.append(train_data[i])
            sus_tags.append(train_labels[i])
for i in range(len(test_data)):
    if 'Sustain' in test_labels[i]:
        previous_lines_sust.append(test_data[i-1])
        sustains.append(test_data[i])
        sus_tags.append(test_labels[i])


# In[18]:


for i in range(len(sus_tags)):
    if 'Append' in sus_tags[i]:
        sus_tags[i]=re.sub('Append','Prolong', sus_tags[i])


# In[19]:


sustains=delete_odds(sustains)


# In[20]:


responds=[]
previous_responds=[]
respond_tags=[]
for i in range(len(train_data)):
    if 'Answer' in train_labels[i]:
        continue
    if 'Disengage' in train_labels[i]:
        continue
    elif 'Respond' in train_labels[i]:
        responds.append(train_data[i])
        previous_responds.append(train_data[i-1])
        respond_tags.append(train_labels[i])


# In[21]:


for i in range(len(test_data)):
    if 'Answer' in test_labels[i]:
        continue
    if 'Disengage' in test_labels[i]:
        continue
    elif 'Respond' in test_labels[i]:
        responds.append(test_data[i])
        previous_responds.append(test_data[i-1])
        respond_tags.append(test_labels[i])


# In[22]:


def clean_responds(respond_tags):
    for i in range(len(respond_tags)):
        if 'Decline' in respond_tags[i]:
            respond_tags[i]=re.sub('Decline','Contradict',respond_tags[i])
        if 'Accept' in respond_tags[i]:
            respond_tags[i]=re.sub('Accept','Affirm',respond_tags[i])
        tag_list=respond_tags[i].split('.')[-2:]
        respond_tags[i]='.'.join(tag_list)
    return respond_tags


# In[23]:


respond_tags=clean_responds(respond_tags)


# In[24]:


previous_lines=[]
replies=[]
tags=[]
for i in range(len(train_data)):
    if 'Reply' in train_labels[i]:
        if '?' not in train_labels[i]:
            previous_lines.append(train_data[i-1])
            replies.append(train_data[i])
            tags.append(train_labels[i])


# In[25]:


for i in range(len(test_data)):
    if 'Reply' in test_labels[i]:
        if '?' not in test_labels[i]:
            previous_lines.append(test_data[i-1])
            replies.append(test_data[i])
            tags.append(test_labels[i])


# In[26]:


for i in range(len(tags)):
    tag_list = tags[i].split('.')[-2:]
    tags[i]=str('.'.join(tag_list))
    if 'Answer' in tags[i]:
        tags[i]='Response.Resolve.'


# In[27]:


train_speakers=dialogues[1][2]
test_speakers = dialogues[0][2]


# In[28]:


train_data = delete_odds(train_data)
test_data = delete_odds(test_data)


# In[29]:


def get_cut_labels(labels):
    for i in range(len(labels)):
        if labels[i].startswith('Open'):
            labels[i] = 'Open.'
        if labels[i].startswith('React.Rejoinder.'):
            labels[i] = 'React.Rejoinder.'
        if labels[i].startswith('React.Respond.'):
            labels[i] = 'React.Respond.'
        if labels[i].startswith('Sustain.Continue.'):
            labels[i] = 'Sustain.Continue.'
    return labels


# In[30]:


cut_train_labels = get_cut_labels(train_labels)


# In[31]:


cut_test_labels = get_cut_labels(test_labels)


# In[32]:


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
embed_model = AutoModel.from_pretrained("DeepPavlov/bert-base-cased-conversational")


# In[236]:


def get_embeddings(data):
    outputs = []
    for text in data:
        with torch.no_grad():
            input_ph= tokenizer(text, padding=True, truncation=True, max_length=30,return_tensors="pt")
            output_ph = embed_model(**input_ph)
    #        train_outputs.append(output_ph.pooler_output.cpu().numpy()) 
            sentence_embedding = output_ph.last_hidden_state.mean(dim=1).cpu().numpy()
        outputs.append(sentence_embedding)
    outputs = np.concatenate(outputs)
    return outputs


# In[237]:


all_outputs=[]
all_outputs.extend(get_embeddings(train_data))
all_outputs.extend(get_embeddings(test_data))


# In[238]:


all_cuts=[]
all_cuts.extend(cut_train_labels)
all_cuts.extend(cut_test_labels)


# In[36]:


# path2 = 'SBC058.json'
# with open(path2) as proba:
#     proba=json.load(proba)
#     dialogue = proba['text']
#     dialogue = [i for i in dialogue if not (i['phrase'] == 'PAUSE')]
#     phrases = []
#     speakers=[]
#     for i in range(len(dialogue)):
#         phrases.append(dialogue[i]['phrase'])
#         speakers.append(dialogue[i]['speaker'])


# In[37]:


phrases = delete_odds(phrases)


# In[38]:


from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(C=0.01,class_weight='balanced')
model.fit(all_outputs,all_cuts)


# In[39]:


def upper_class_predict(phrase_embeddings, model):
    y_pred_sample = model.predict(phrase_embeddings)
    return y_pred_sample


# # Questions

# In[49]:


interrogative_words = ['whose', 'what', 'which', 'who', 'whom', 'what', 'which','why', 'where', 'when', 'how']


# Detect all questions in phrases

# In[50]:


with open('track_list') as track_list:
    track_list= track_list.readlines()
train_que =[]
train_tags=[]
for line in track_list:
    line = line.split('/')
    train_que.append(line[0])
    train_tags.append(line[1][:-1])


# In[51]:


train_em_que = get_embeddings(train_que)


# In[52]:


lr2 = LogisticRegression(C=0.01, class_weight='balanced', solver="newton-cg")
lr2.fit(train_em_que,train_tags)


# In[53]:


true_tracks = {'1':'Track.Check', '2':'Track.Confirm', '3':'Track.Clarify','4':'Track.Probe'}


# In[54]:


def map_tracks(tags_for_track, true_tracks):
    for i in range(len(list(tags_for_track))):
        if tags_for_track[i]=='5':
            tag=tags_for_track[i]
        else:
            tag=true_tracks[tags_for_track[i]]
    return tag


# In[55]:


map_tracks(['1'],true_tracks)


# In[56]:


def get_label_for_question(phrase,y_pred,current_speaker, previous_speaker, true_tracks, model,interrogative_words):
    questions=[]
    questions.append(phrase)
    question_embeddings = get_embeddings(questions)
    y_pred_track = lr2.predict(question_embeddings)
#     print(y_pred_track)
    tag_for_track=map_tracks(y_pred_track,true_tracks)
#     print(tag_for_track)
    if current_speaker!=previous_speaker:
        if y_pred=='React.Respond.' and tag_for_track!='5':
            y_pred='React.Rejoinder.Support.'+tag_for_track
        if y_pred=='React.Rejoinder.' and tag_for_track!='5':
            y_pred+='Support.'
            y_pred+=tag_for_track
        if y_pred=='React.Rejoinder.' and tag_for_track=='5':
            for word in interrogative_words:
                if word in phrase:
                    y_pred='React.Rejoinder.Confront.Challenge.Rebound'
                else:
                    y_pred='React.Rejoinder.Confront.Response.Re-challenge'
        if y_pred=='Sustain.Continue.':
            y_pred='React.Rejoinder.Support.'+tag_for_track
    if current_speaker==previous_speaker:
        if  y_pred=='React.Rejoinder.' and tag_for_track!='5':
            y_pred+='Support.'
            y_pred+=tag_for_track
        if y_pred=='React.Rejoinder.' and tag_for_track=='5':
            for word in interrogative_words:
                if word in phrase:
                    y_pred='React.Rejoinder.Confront.Challenge.Rebound'
                else:
                    y_pred='React.Rejoinder.Confront.Response.Re-challenge'

        if y_pred=='Sustain.Continue.':
            y_pred='Sustain.Continue.Monitor'
    return y_pred


# In[297]:


print(get_label_for_question('Do you?','React.Rejoinder.','SHER', 'SHER',true_tracks,lr2, interrogative_words))


# # Sustain

# In[58]:


# for sustain
def check_develop(phrase,previous_phrase,y_pred, y_pred_previous,current_speaker, previous_speaker):
    if current_speaker!=previous_speaker:
        if 'Sustain.Continue.' in y_pred_previous:
            if 'Sustain.Continue.' in y_pred:
                y_pred='React.Respond.Support.Develop.'
    return y_pred


# In[59]:


# for every line
def check_sustain(phrase,y_pred,y_pred_previous,current_speaker, previous_speaker):
    if current_speaker==previous_speaker:
        if y_pred_previous=='Sustain.Continue.':
             if y_pred!='Open.':
                y_pred='Sustain.Continue.'
    return y_pred


# In[60]:


# if y_pred[i]=='Sustain.Continue.' or y_pred[i]=='React.Respond.Develop.':
def get_label_for_sustains(phrase, y_pred, sus_model):
    test_sustains=[]
    test_sustains.append(phrase)
    test_sustains_emb=get_embeddings(test_sustains)
    tags_for_sus=sus_model.predict(test_sustains_emb)
    if y_pred=='Sustain.Continue.':
        y_pred=''.join(tags_for_sus)
    if y_pred=='React.Respond.Support.Develop.':
        cut_tags=''.join(tags_for_sus).split('.')[-1]
        if cut_tags!='Monitor':
            y_pred+=cut_tags
    return y_pred


# In[61]:


train_sustains=get_embeddings(sustains)


# In[62]:


lr_sus = LogisticRegression(C=0.01, class_weight='balanced', solver="newton-cg")
lr_sus.fit(train_sustains,sus_tags)


# In[63]:


get_label_for_sustains("And I think it's a great thing","Sustain.Continue.", lr_sus)


# # Replies

# In[182]:


train_embed_replies=get_embeddings(replies)


# In[183]:


train_prev_lines=get_embeddings(previous_lines)


# In[184]:


reply_concatenate = np.concatenate([train_embed_replies,train_prev_lines], axis=1)


# In[254]:


set(respond_tags)


# In[186]:


from sklearn.linear_model import LogisticRegression
lr_reply = LogisticRegression(C=0.01, class_weight='balanced',solver="newton-cg")
lr_reply.fit(reply_concatenate,tags)


# In[187]:


train_emb_responds=get_embeddings(responds)


# In[188]:


train_prev_responds=get_embeddings(previous_responds)


# In[189]:


responds_concatenate = np.concatenate([train_emb_responds,train_prev_responds], axis=1)


# In[190]:


from sklearn.preprocessing import Normalizer
# responds_X = Normalizer().fit_transform(responds_concatenate)
lr_responds = LogisticRegression(C=0.5, class_weight='balanced', solver="newton-cg")
lr_responds.fit(responds_concatenate,respond_tags)


# In[191]:


from spacy import morphology 


# In[255]:


def get_label_for_responds(phrase,previous_phrase, y_pred,y_pred_previous,current_speaker,previous_speaker, model1, model2):
    confront_labels=['Reply.Disawow','Reply.Disagree','Reply.Contradict']
    support_labels=['Reply.Acknowledge','Reply.Affirm','Reply.Agree', 'Develop.Elaborate', 'Develop.Enhance',
 'Develop.Extend']
    try_replies=[]
    test_prev_lines = []
    test_responds=[]
    test_responds_prev_lines = []
    if '?' in previous_phrase:
        if current_speaker!=previous_speaker:
            test_prev_lines.append(previous_phrase)
            try_replies.append(phrase)
            test_concat = np.concatenate([get_embeddings(try_replies),get_embeddings(test_prev_lines)],axis=1)
            tag_for_reply = model1.predict(test_concat)
            if tag_for_reply=='Reply.Decline':
                tag_for_reply='Reply.Contradict'
            if tag_for_reply in confront_labels:
                y_pred=y_pred+'Confront.'+''.join(tag_for_reply)
            if 'Response.Resolve.' in tag_for_reply:
                y_pred='React.Rejoinder.Support.Response.Resolve'
            else:
                y_pred=y_pred+'Support.'+''.join(tag_for_reply)
            if 'yes' in str(try_replies).lower():
                if y_pred=='React.Respond.Reply.Disagree':
                    if 'Confirm' in y_pred_previous:
                        return 'React.Respond.Support.Reply.Affirm'
                
                
        else:
            test_responds_prev_lines.append(previous_phrase)
            test_responds.append(phrase)
            test_responds_concatenate = np.concatenate([get_embeddings(test_responds),get_embeddings(test_responds_prev_lines)], axis=1)
            tags_for_responds = model2.predict(test_responds_concatenate)
            print(tags_for_responds)
            for token in nlp(phrase):
                if token.dep_=='neg':
                    return 'React.Rejoinder.Confront.Challenge.Counter'
            if tags_for_responds in support_labels:
                return y_pred+'Support.'+''.join(tags_for_responds)
            if tags_for_responds in confront_labels:
                return y_pred+'Confront.'+''.join(tags_for_responds)
            else:
                return y_pred+''.join(tags_for_responds)

                
    else:
        test_responds_prev_lines.append(previous_phrase)
        test_responds_concatenate = np.concatenate([get_embeddings(test_responds),get_embeddings(test_responds_prev_lines)], axis=1)
        tags_for_responds = model2.predict(test_responds_concatenate)
        print(tags_for_responds)
        for token in nlp(phrase):
            if token.dep_=='neg':
                return 'React.Rejoinder.Confront.Challenge.Counter'
        if tags_for_responds in support_labels:
            return y_pred+'Support.'+''.join(tags_for_responds)
        if tags_for_responds in confront_labels:
            return y_pred+'Confront.'+''.join(tags_for_responds)
        else:
            return y_pred+''.join(tags_for_responds)


# In[ ]:


train_embed_replies = get_embeddings(replies)
train_prev_lines = get_embeddings(previous_lines)
reply_concatenate = np.concatenate([train_embed_replies, train_prev_lines], axis=1)

lr_reply = LogisticRegression(C=0.01, class_weight="balanced", solver="newton-cg")
lr_reply.fit(reply_concatenate, tags)

train_emb_responds = get_embeddings(responds)
train_prev_responds = get_embeddings(previous_responds)
responds_concatenate = np.concatenate([train_emb_responds, train_prev_responds], axis=1)

lr_responds = LogisticRegression(C=0.5, class_weight="balanced", solver="newton-cg")
lr_responds.fit(responds_concatenate, respond_tags)


# In[276]:


def get_label_for_responds(phrase, previous_phrase, y_pred, y_pred_previous, current_speaker, previous_speaker):
    confront_labels=['Reply.Disawow','Reply.Disagree','Reply.Contradict']
    support_labels=['Reply.Acknowledge','Reply.Affirm','Reply.Agree', 'Develop.Elaborate', 'Develop.Enhance',
 'Develop.Extend']
    try_replies = []
    test_prev_lines = []
    test_responds = []
    test_responds_prev_lines = []
    if "?" in previous_phrase:
        if current_speaker != previous_speaker:
            test_prev_lines.append(previous_phrase)
            try_replies.append(phrase)
            test_concat = np.concatenate([get_embeddings(try_replies), get_embeddings(test_prev_lines)], axis=1)
            tag_for_reply = lr_reply.predict(test_concat)
            if tag_for_reply=='Reply.Decline':
                tag_for_reply='Reply.Contradict'
            if "yes" in str(try_replies).lower():
                if "Disagree" in tag_for_reply:
                    if "Confirm" in y_pred_previous:
                        y_pred = "React.Respond.Support.Reply.Affirm"
            if tag_for_reply in confront_labels:
                y_pred=y_pred+'Confront.'+''.join(tag_for_reply)
            if tag_for_reply in support_labels:
                y_pred=y_pred+'Support.'+''.join(tag_for_reply)
            if 'Response.Resolve.' in tag_for_reply:
                y_pred='React.Rejoinder.Support.Response.Resolve'
                

        else:
            test_responds_prev_lines.append(previous_phrase)
            test_responds.append(phrase)
            test_responds_concatenate = np.concatenate(
                [get_embeddings(test_responds), get_embeddings(test_responds_prev_lines)], axis=1
            )
            tags_for_responds = lr_responds.predict(test_responds_concatenate)
            if tags_for_responds in confront_labels:
                y_pred=y_pred+'Confront.'+''.join(tags_for_responds)
            if tags_for_responds in support_labels:
                y_pred=y_pred+'Support.'+''.join(tags_for_responds)
            for token in nlp(phrase):
                if token.dep_=='neg':
                    return 'React.Rejoinder.Confront.Challenge.Counter'

    else:
        test_responds_prev_lines.append(previous_phrase)
        test_responds.append(phrase)
        test_responds_concatenate = np.concatenate(
            [get_embeddings(test_responds), get_embeddings(test_responds_prev_lines)], axis=1
        )
        tags_for_responds = lr_responds.predict(test_responds_concatenate)
        if tags_for_responds in support_labels:
            y_pred=y_pred+'Support.'+''.join(tags_for_responds)
        elif tags_for_responds in confront_labels:
            y_pred=y_pred+'Confront.'+''.join(tags_for_responds)
        else:
            y_pred=y_pred+''.join(tags_for_responds)
        for token in nlp(phrase):
            if token.dep_=='neg':
                y_pred='React.Rejoinder.Confront.Challenge.Counter'
    return y_pred


# In[295]:


print(get_label_for_responds("Absolutely not","Is she  great?", 'React.Respond.','Monitor' ,'1','0'))


# # Rejoinder

# In[288]:


def get_labels_for_rejoinder(phrase, previous_phrase, current_speaker, previous_speaker):
    for token in nlp(phrase):
        if token.dep_=='neg':
            y_pred = "React.Rejoinder.Confront.Challenge.Counter"
            return y_pred
    if "?" in previous_phrase:
        if previous_speaker != current_speaker:
            y_pred = "React.Rejoinder.Support.Response.Resolve"
        else:
            y_pred = "React.Rejoinder.Confront.Response.Re-challenge"
    else:
        y_pred = "React.Respond.Support.Develop.Extend"
    return y_pred


# # Open

# In[92]:


def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close() 
    return file

def divide_into_sentences(document):
    return [sent for sent in document.sents]


# In[93]:


def number_of_fine_grained_pos_tags(sent):
    tag_dict = {'-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0, 
    '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
    'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 
    'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 
    'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 
    'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 
    'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
    'OOV': 0, 'TRAILING_SPACE': 0}
    for token in sent:
        if token.is_oov:
            tag_dict['OOV']+= 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE']+= 1
        else:
            tag_dict[token.tag_]+= 1
            
    return tag_dict


# In[94]:


def number_of_dependency_tags(sent):
    dep_dict = {'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
    'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
    'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
    'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
    'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
    'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
    'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
    'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
    'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0, 
    'TRAILING_DEP': 0}
    
    for token in sent:
        if token.dep_== '':
            dep_dict['TRAILING_DEP']+= 1
        else:
            try:
                dep_dict[token.dep_]+= 1
            except:
                print('Unknown dependency for token: "' + token.orth_ +'". Passing.')
        
    return dep_dict


# In[95]:


def number_of_specific_entities(sent):
    entity_dict = {
    'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
    'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
    'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
    'ORDINAL': 0, 'CARDINAL': 0 }    
    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity]+=1     
    return entity_dict


# In[96]:


def predict(test_sent, classifier, scaler=None):
    parsed_test = divide_into_sentences(nlp(test_sent))
    # Get features
    sentence_with_features = {}
    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)
    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)
    #dep_dict = number_of_dependency_tags(parsed_test[0])
    #sentence_with_features.update(dep_dict)
    df = pd.DataFrame(sentence_with_features, index=[0])
    if scaler:
        df = scaler.transform(df)
    
    prediction = classifier.predict(df)
    if prediction == 0:
        open_tag='Fact'
    else:
        open_tag='Opinion'
    return open_tag


# In[97]:


nn_classifier = load_pickle('nn_classifier.pickle')


# In[98]:


scaler = load_pickle('scaler.pickle')


# In[99]:


# if y_pred[i]=='Open.':
def get_open_labels(phrase,y_pred,nn_classifier,scaler):
    open_tag =predict(phrase,nn_classifier,scaler)
    if open_tag=='Fact':
        if '?' not in phrase:
            open_tag='Give.Fact'
        else:
            open_tag='Demand.Fact'
    else:
        if '?' not in phrase:
            open_tag='Give.Opinion'
        else:
            open_tag='Demand.Opinion'
    y_pred=y_pred+open_tag
    if len(word_tokenize(phrase))<4:
        poses=[]
        doc=nlp(phrase)
        for token in doc:
            poses.append(token.pos_)
        if 'PROPN' in poses:
            y_pred='Open.Attend'
    return y_pred            


# In[284]:


def classify_labels(phrases,speakers):
    y_preds=[]
    for i in range(len(phrases)):
        test_embeddings=get_embeddings([phrases[i]])
        y_pred=''.join(list(upper_class_predict(test_embeddings, model)))
        if i==0:
            y_pred='Open.'
            y_pred=get_label_for_sustains(phrases[i], y_pred, lr_sus)
            y_preds.append(y_pred)
        else:
            y_pred=check_sustain(phrases[i],y_pred,y_preds[i-1],speakers[i], speakers[i-1])
            if y_pred=='Open.':
                pass
            if y_pred=='Sustain.Continue.':
                y_pred=check_develop(phrases[i],phrases[i-1],y_pred, y_preds[i-1],speakers[i], speakers[i-1])
                y_pred=get_label_for_sustains(phrases[i], y_pred, lr_sus)
            if '?' in phrases[i]:
                y_pred=get_label_for_question(phrases[i],y_pred,speakers[i], speakers[i-1], true_tracks, lr2,interrogative_words)
            if y_pred=='React.Respond.':
                 y_pred=get_label_for_responds(phrases[i],phrases[i-1], y_pred,y_preds[i-1],speakers[i],speakers[i-1])
            if y_pred=='React.Rejoinder.':
                y_pred=get_labels_for_rejoinder(phrases[i],y_pred,y_preds[i-1],phrases[i-1], speakers[i], speakers[i-1])
            y_preds.append(y_pred)
    return y_preds


# In[286]:


def get_speech_function(phrase, prev_phrase, prev_speech_function, speaker="John", previous_speaker="Doe"):
    # note: default values for current and previous speaker are only to make them different. In out case they are always
    # different (bot and human)
    test_embeddings = get_embeddings([phrase])
    y_pred = "".join(list(upper_class_predict(test_embeddings, model)))
    if prev_phrase is None:
        y_pred = get_open_labels(phrase, "Open.")
    else:
        if y_pred == "Open.":
            y_pred = get_open_labels(phrase, y_pred)
        if y_pred == "Sustain.Continue.":
            y_pred = check_develop(y_pred, prev_speech_function, speaker, previous_speaker)
            y_pred = get_label_for_sustains(phrase, y_pred)
        if "?" in phrase:
            y_pred = get_label_for_question(phrase, y_pred, speaker, previous_speaker)
        if y_pred == "React.Respond.":
            y_pred = get_label_for_responds(
                phrase, prev_phrase, y_pred, prev_speech_function, speaker, previous_speaker
            )
        if y_pred == "React.Rejoinder.":
            y_pred = get_labels_for_rejoinder(phrase, prev_phrase, speaker, previous_speaker)
    return y_pred


# In[287]:


y_preds


# In[129]:


phrases_to_check=['fine, thank you.', 'and you?']


# In[130]:


speakers_tocheck=['0','1','1']


# In[124]:


for i in zip(speakers_tocheck, phrases_to_check, y_preds[1:]):
    print(i)


# In[180]:


len(y_preds)


# In[103]:


def get_speech_function(phrase, prev_phrase, prev_speech_function, speaker='John', previous_speaker='Doe'):
    # note: default values for current and previous speaker are only to make them different. In out case they are always
    # different (bot and human)
    test_embeddings = get_embeddings([phrase])
    y_pred = ''.join(list(upper_class_predict(test_embeddings, model)))
    if prev_phrase is None:
        y_pred = get_open_labels(phrase, 'Open.')
    else:
        if y_pred == 'Open.':
            y_pred = get_open_labels(phrase, y_pred)
        if y_pred == 'Sustain.Continue.':
            y_pred = check_develop(y_pred, prev_speech_function, speaker, previous_speaker)
            y_pred = get_label_for_sustains(phrase, y_pred)
        if '?' in phrases[i]:
            y_pred = get_label_for_question(phrase, y_pred, speaker, previous_speaker)
        if y_pred == 'React.Respond.':
            y_pred = get_label_for_responds(phrase, prev_phrase, y_pred, prev_speech_function, speaker,
                                            previous_speaker)
        if y_pred == 'React.Rejoinder.':
            y_pred = get_labels_for_rejoinder(phrase, prev_phrase, speaker, previous_speaker)
    return y_pred


# In[80]:


get_speech_function('and you?','How are you doing today?')


# In[134]:


get_label_for_question('and you?', 'React.Rejoinder.', 'speaker', 'speaker',true_tracks,lr2, interrogative_words)


# In[98]:


from nltk import sent_tokenize


# In[100]:


sent_tokenize('and you?')


# In[ ]:




