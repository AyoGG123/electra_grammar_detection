# -*- coding: UTF-8 -*-
import time
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from app.prediction import predict
import torch
import random
import sys
from transformers import ElectraForSequenceClassification, ElectraTokenizer

ROOT = os.path.abspath(os.path.dirname(os.getcwd()))
print(ROOT)
sent_delimiter = ['，', '。', '？', '！', '；']

# print(f'using {device} device')
app = Flask(__name__)
CORS(app)
time_start = 0

'''#初始化並載入模型
electra_tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
ElectraClassifier = ElectraForSequenceClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
ElectraClassifier.to(device)
checkpoint = torch.load(f'app/useModel/ELECTRA_binary_seq_cls_cnv0.4nlpcc18_seq_10.h5',map_location=device)
ElectraClassifier.load_state_dict(checkpoint)
ElectraClassifier.eval()'''

electra_tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
ElectraClassifier = ElectraForSequenceClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if sys.platform == 'darwin': device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

ElectraClassifier = torch.load(os.path.join('app', 'useModel', 'ELECTRA_binary_seq_cnv0.8.1_20.h5'),
                               map_location=device)
ElectraClassifier.to(device)
ElectraClassifier.eval()

correct = []
error = []

with open(os.path.join('app', 'static', 'example.txt'), 'r', encoding='utf-8') as f:
    data_ = f.readlines()
    for line in data_:
        line = line.split("^")
        if '0' in line[1]:
            correct.append(line[0])
        else:
            error.append(line[0])


@app.route('/test', methods=['GET'])
def getResult():
    return 'hey hey'


@app.route('/predict', methods=['POST'])
def postInput():
    global time_start
    time_start = time.time()
    insertValues = request.get_json()
    # print(insertValues)  # {'chunks': ['很多人被别人否定，就放弃自己的梦。'], 'chinese': '簡體中文'}
    result, prob = Input(insertValues)
    print(f"prob={prob}")
    print(f"全形={halfwidth_to_fullwidth(insertValues['chunks'][0])}")
    # print(result) #[['1', '两年前，我发现小康患有重度龋齿，四颗门牙都是黑黑的，只剩下牙根。', '1']]

    return jsonify({"return": result})


@app.route('/correct_example', methods=['POST'])
def post_correct_example():
    # 在这里执行您的处理逻辑，然后返回相应的数据
    correct_example = {"correct_example": [correct[random.randint(0, len(correct) - 1)].strip(),
                                           correct[random.randint(0, len(correct) - 1)].strip()]}
    return jsonify(correct_example)


@app.route('/error_example', methods=['POST'])
def post_error_example():
    # 在这里执行您的处理逻辑，然后返回相应的数据
    error_example = {"error_example": [error[random.randint(0, len(error) - 1)].strip(),
                                       error[random.randint(0, len(error) - 1)].strip()]}
    return jsonify(error_example)


def halfwidth_to_fullwidth(input_str):
    result_str = ""
    for char in input_str:
        code = ord(char)
        # 如果是全角空格，直接转换为半角空格
        if code == 12288:
            result_str += chr(32)
        # 如果是全角字符，转换为半角字符
        elif 'Ａ' <= char <= 'Ｚ':
            result_str += chr(ord(char) - ord('Ａ') + ord('A'))
        elif 'ａ' <= char <= 'ｚ':
            result_str += chr(ord(char) - ord('ａ') + ord('a'))
        elif '0' <= char <= '9':
            result_str += chr(code)
        # 如果是全角标点符号，转换为相应的全角标点符号
        elif char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
            result_str += chr(ord(char) + 65248)
        # 其他字符保持不变
        else:
            result_str += chr(code)

    return result_str


def module_setting(data):
    encoded_sent = electra_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=data,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True
    )

    # Convert lists to tensors
    input_ids = torch.tensor(encoded_sent['input_ids'])
    attention_masks = torch.tensor(encoded_sent['attention_mask'])

    return input_ids, attention_masks


def get_test_dataloader(inputs, attention_mask):
    expand_dim = lambda tlist, x: [tlist[i:i + x] for i in range(0, len(tlist), x)]
    inputs = expand_dim(inputs, 64)
    attention_mask = expand_dim(attention_mask, 64)
    dataset = []
    for i, k in zip(inputs, attention_mask):
        dataset.append(tuple((i, k)))
    return dataset


def Input(insertValues):
    sentences = insertValues['chunks']
    chinese = insertValues['chinese']
    result = []
    num = 0
    sentences = list(map(lambda x: x + '。' if x[-1] not in sent_delimiter else x, sentences))  # 如果最後一個字不是標點符號，就加上句號
    test_inputs, test_attention_mask = module_setting(sentences)
    test_dataloader = get_test_dataloader(test_inputs, test_attention_mask)
    for i, batch in enumerate(test_dataloader):
        id, sentence, predict_, num, prob = predict(batch, ElectraClassifier, electra_tokenizer, num, device)
        result.append(id)
        result.append(sentence)
        result.append(predict_)
        # result.append(prob)
        prob_ = prob

    output = []
    for i, j, k in zip(result[0], result[1], result[2]):
        output.append([i, j, k])

    return output, prob_


def print_time():
    global time_start
    time_end = time.time()
    time_cost = time_end - time_start
    h = int(time_cost // 60 // 60)
    m = int(time_cost // 60 % 60)
    s = int(time_cost % 60)
    print(f'time cost : {h}h {m}m {s}s')


if __name__ == '__main__':
    print(halfwidth_to_fullwidth('我们三个人一家要生活在一起的。'))
