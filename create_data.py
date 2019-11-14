import numpy as np
import pandas as pd
import os
import json
import gc
from tqdm import tqdm
import re
for dirname, _, filenames in os.walk('input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = 'input/'
train_path = 'simplified-nq-train.jsonl'
test_path = 'simplified-nq-test.jsonl'
sample_submission_path = 'sample_submission.csv'
html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>', \
             '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'of', 'in', 'and', 'on', \
         'what', 'where', 'when', 'which'] + html_tags

def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, '')
    x = re.sub(' +', ' ', x)
    return x

def read_data(path,sample=False,chunksize=500):
    if sample==True:
        df = []
        with open(path, 'rt') as reader:
            for i in range(chunksize):
                df.append(json.loads(reader.readline()))
        df = pd.DataFrame(df)
        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    else:
        df=[]
        with open(path,'r') as f:
            for line in tqdm(f):
                line = json.loads(line)
                df.append(line)
        df=pd.DataFrame(df)
        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    return df

train = read_data(path+train_path)
test = read_data(path+test_path)

def extract_target_variable(df, short = True):
    if short:
        short_answer = []
        for i in tqdm(range(len(df))):
            short = df['annotations'][i][0]['short_answers']
            if short == []:
                yes_no = df['annotations'][i][0]['yes_no_answer']
                if yes_no == 'NO' or yes_no == 'YES':
                    short_answer.append(yes_no)
                else:
                    short_answer.append('EMPTY')
            else:
                short = short[0]
                st = short['start_token']
                et = short['end_token']
                short_answer.append(f'{st}'+':'+f'{et}')
        short_answer = pd.DataFrame({'short_answer': short_answer})
        return short_answer
    else:
        long_answer = []
        for i in tqdm(range(len(df))):
            long = df['annotations'][i][0]['long_answer']
            if long['start_token'] == -1:
                long_answer.append('EMPTY')
            else:
                st = long['start_token']
                et = long['end_token']
                long_answer.append(f'{st}'+':'+f'{et}')
        long_answer = pd.DataFrame({'long_answer': long_answer})
        return long_answer

def build_train_test_long(df, train = True):
    final_long_answer_frame = pd.DataFrame()
    if train == True:
        # get long answer
        long_answer = extract_target_variable(df, False)
        
        # iterate over each row to get the possible answers
        for index, row in tqdm(df.iterrows()):
            start_end_tokens = []
            questions = []
            responds = []
            for i in row['long_answer_candidates']:
                start_token = i['start_token']
                end_token = i['end_token']
                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])
                question = clean(row['question_text'])
                respond = clean(" ".join(row['document_text'].split()[start_token : end_token]))
                start_end_tokens.append(start_end_token)
                questions.append(question)
                responds.append(respond)

            long_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})
            long_answer_frame['answer'] = long_answer.iloc[index][0]
            long_answer_frame['target'] = long_answer_frame['start_end_token'] == long_answer_frame['answer']
            long_answer_frame['target'] = long_answer_frame['target'].astype('int16')
            long_answer_frame.drop(['answer'], inplace = True, axis = 1)
            final_long_answer_frame = pd.concat([final_long_answer_frame, long_answer_frame])
        return final_long_answer_frame
    else:
         # iterate over each row to get the possible answers
        for index, row in df.iterrows():
            start_end_tokens = []
            questions = []
            responds = []
            for i in row['long_answer_candidates']:
                start_token = i['start_token']
                end_token = i['end_token']
                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])
                question = row['question_text']
                respond = " ".join(row['document_text'].split()[start_token : end_token])
                start_end_tokens.append(start_end_token)
                questions.append(question)
                responds.append(respond)

            long_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})
            final_long_answer_frame = pd.concat([final_long_answer_frame, long_answer_frame])
        return final_long_answer_frame
        


def build_train_test_short(df, train = True):
    
    final_short_answer_frame = pd.DataFrame()
    
    if train == True:
        # get short answer
        short_answer = extract_target_variable(df, True)

        # iterate over each row to get the possible answer
        for index, row in df.iterrows():
            start_tokens = []
            end_tokens = []
            start_end_tokens = []
            questions = []
            responds = []
            for i in row['long_answer_candidates']:
                start_token = i['start_token']
                end_token = i['end_token']
                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])
                question = row['question_text']
                respond = " ".join(row['document_text'].split()[int(start_token) : int(end_token)])
                start_tokens.append(start_token)
                end_tokens.append(end_token)
                start_end_tokens.append(start_end_token)
                questions.append(question)
                responds.append(respond)

            short_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_token': start_tokens, 'end_token': end_tokens, 'start_end_token': start_end_tokens})
            short_answer_frame['answer'] = short_answer.iloc[index][0]
            short_answer_frame['start_token_an'] = short_answer_frame['answer'].apply(lambda x: x.split(':')[0] if ':' in x else 0)
            short_answer_frame['end_token_an'] = short_answer_frame['answer'].apply(lambda x: x.split(':')[1] if ':' in x else 0)
            short_answer_frame['start_token_an'] = short_answer_frame['start_token_an'].astype(int)
            short_answer_frame['end_token_an'] = short_answer_frame['end_token_an'].astype(int)
            short_answer_frame['target'] = 0
            short_answer_frame.loc[(short_answer_frame['start_token_an'] >= short_answer_frame['start_token']) & (short_answer_frame['end_token_an'] <= short_answer_frame['end_token']), 'target'] = 1
            short_answer_frame.drop(['answer', 'start_token', 'end_token', 'start_token_an', 'end_token_an'], inplace = True, axis = 1)
            final_short_answer_frame = pd.concat([final_short_answer_frame, short_answer_frame])
        return final_short_answer_frame
    else:
        # iterate over each row to get the possible answer
        for index, row in df.iterrows():
            start_end_tokens = []
            questions = []
            responds = []
            for i in row['long_answer_candidates']:
                start_token = i['start_token']
                end_token = i['end_token']
                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])
                question = row['question_text']
                respond = " ".join(row['document_text'].split()[int(start_token) : int(end_token)])
                start_end_tokens.append(start_end_token)
                questions.append(question)
                responds.append(respond)

            short_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})
            final_short_answer_frame = pd.concat([final_short_answer_frame, short_answer_frame])
        return final_short_answer_frame

sh = build_train_test_long(train)
sh.to_csv("train_long.csv",index=None)