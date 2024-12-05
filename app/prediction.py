# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import re


def predict(batch, bert_classifier, electra_tokenizer, num, device):  # sentence檢測
    output = {'id': [], 'chunks': [], 'pred': [], 'prob': []}
    sent_delimiter_ids = [8024, 511, 8043, 8013, 8039]
    # ['，', '。', '？', '！', '；']

    for input_ids, attention_mask in zip(*tuple(t.to(device) for t in batch)):
        s_input_ids, s_attention_mask = split_chunk_into_sent(input_ids, attention_mask, sent_delimiter_ids)

        s_input_ids = s_input_ids.to(device)
        s_attention_mask = s_attention_mask.to(device)

        with torch.no_grad():
            logits = bert_classifier(input_ids=s_input_ids, attention_mask=s_attention_mask)

        pred_result = torch.argmax(logits[0], dim=1).tolist()
        prob = F.softmax(logits[0], dim=-1).tolist()
        t = electra_tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
        num += 1
        temp = re.sub(r' ', '', t)

        for line in re.split("，|。|？|！|；", temp):
            output['chunks'].append(line)

        output['chunks'] = output['chunks'][:-1]
        output['pred'] = pred_result
        output['prob'] = prob

        for i in range(len(output['pred'])):
            output['id'].append(str(i + 1))

        for i in range(len(output['pred'])):
            # print(output['pred'][i])
            # print(type(output['pred'][i]))
            output['pred'][i] = '錯誤' if output['pred'][i] == 1 else '正確'
    return output['id'], output['chunks'], output['pred'], num, output['prob']


# def predict(batch, bert_classifier, electra_tokenizer, num, device):  # chunk檢測
#     result = {'id': [], 'sentences': [], 'truth': [], 'pred': [], 'prob': []}
#
#     b_input_ids, b_attention_mask = batch
#     b_input_ids = b_input_ids.to(device)
#     b_attention_mask = b_attention_mask.to(device)
#
#     with torch.no_grad():
#         output = bert_classifier(input_ids=b_input_ids, attention_mask=b_attention_mask)
#         logits = output.logits
#
#     text = electra_tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
#     text = [list(filter(lambda x: x != '[PAD]', x.split(' '))) for x in text]
#
#     for i in text:
#         if '[CLS]' in i: i.remove('[CLS]')
#         if '[SEP]' in i: i.remove('[SEP]')
#
#     for logit, t, _ids in zip(logits, text, b_input_ids):
#         pred = logit.argmax().item()
#         prob = F.softmax(logit, dim=-1).tolist()
#         # threshold = 0.97
#         # pred = 1 if prob[1]>threshold else 0
#         num += 1
#         result['id'].append(str(num))
#         result['sentences'].append(''.join(t))
#         result['pred'].append('錯誤' if '1' in str(pred) else '正確')
#         result['prob'].append(str(prob))
#
#     return result['id'], result['sentences'], result['pred'], num, result['prob']


# 分割標點符號
def split_chunk_into_sent(input_ids, attention_mask, sent_delimiter_ids):
    s_input_ids = []
    s_attention_mask = []
    start = 0
    assert len(input_ids) == len(attention_mask)

    for inx, (id, att) in enumerate(zip(input_ids, attention_mask)):
        if att == 0: break
        if id in sent_delimiter_ids:
            input_id = [101] + input_ids[start:inx + 1].tolist() + [102] + [0 for _ in range(
                512 - 2 - len(input_ids[start:inx + 1]))]
            input_id = torch.tensor(input_id)
            s_input_ids.append(input_id)
            s_attention_mask.append(torch.tensor([1] + attention_mask[start:inx + 1].tolist() + [1] + [0 for _ in range(
                512 - 2 - len(attention_mask[start:inx + 1]))]))
            start = inx + 1

    return torch.stack(s_input_ids), torch.stack(s_attention_mask)


if __name__ == "__main__":
    pass
