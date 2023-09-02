
from tqdm import tqdm

import torch
import torchtext
from torchtext.data.metrics import bleu_score

langs = ['en', 'fr', 'de', 'it']

def sent2tensor(tokenize_en, src_field, trg_field, device, max_len, sentence=None):
  if sentence != None:
    if isinstance(sentence, str):
      tokens = tokenize_en(sentence)
    else:
      tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq_len, N] w/ N=1 for batch
    src_len_tensor = torch.LongTensor([len(src_indexes)]).to(device)
    return src_tensor, src_len_tensor

  trg_tensor = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]] + [0 for i in range(1, max_len)]).view(-1, 1).to(device) # [seq_len, 1]
  trg_len_tensor = torch.LongTensor([max_len]).to(device)
  return trg_tensor, trg_len_tensor

def idx2sent(trg_field, arr):
  n_sents = arr.shape[1]  # arr = [seq_len, N]
  results = []
  for i in range(n_sents):  # for each sent
    pred_sent = []
    pred = arr[:, i]
    for i in pred[1:]:  # for each word
      pred_sent.append(trg_field.vocab.itos[i])
      if i == trg_field.vocab.stoi[trg_field.eos_token]: break
    results.append(pred_sent)
  return results

def translate_sentence(sentence, tokenize_en, src_field, trg_field, model, model_cfg, device, src_lang, trg_lang, max_len=64):
    model.eval()
    with torch.no_grad():
      # get data
      src_tensor, src_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len, sentence)
      trg_tensor, trg_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len)
      data = {src_lang: (src_tensor, src_len_tensor)}
      for l in langs:
        if l == src_lang: continue
        data[l] = (trg_tensor.detach().clone(), trg_len_tensor.detach().clone())
      # feed model
      output, _ = model(data, model_cfg, None, 0) # output = [trg_len, N, dec_emb_dim] w/ N=1
      output = output.argmax(-1).detach().cpu().numpy() # output = [seq_len, N]
      results = idx2sent(trg_field, output)[0]
      return results[:-1] # remove <eos>

def translate_batch(model, model_cfg, iterator, tokenize_en, src_field, trg_field, device, src_lang, trg_lang, max_len=64, batch_lim=None):
  model.eval()
  if torchtext.__version__ == '0.6.0': isToDict=True
  with torch.no_grad():
    # x_sents = []
    gt_sents, pred_sents = [], []
    for idx, batch in enumerate(tqdm(iterator)):
      # get data
      if isToDict: batch = vars(batch)
      src = batch[src_lang] # (data, seq_len)
      trg = batch[trg_lang]

      _, N = src[0].shape
      trg_tensor, trg_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len)
      trg_datas = torch.cat([trg_tensor for _ in range(N)], dim=1)
      trg_lens = torch.cat([trg_len_tensor for _ in range(N)], dim=0)

      data = {src_lang: src}
      for l in langs:
        if l == src_lang: continue
        data[l] = (trg_datas.detach().clone(), trg_lens.detach().clone())

      # feed model
      output, _ = model(data, model_cfg, None, 0)
      pred = output.argmax(-1) # [seq_len, N]

      # x_sents = x_sents + idx2sent(src_field, src[0])
      pred_sents.extend(idx2sent(trg_field, pred))
      gt_sents.extend(idx2sent(trg_field, trg[0]))

      if batch_lim!=None and idx==batch_lim: break
    pred_sents = [sent[:-1] for sent in pred_sents]
    gt_sents = [sent[:-1] for sent in gt_sents]
    return pred_sents, gt_sents

def calculate_bleu_batch(model, model_cfg, iterator, tokenizer_en, src_field, trg_field, device, src_lang, trg_lang, max_len=64):
  pred_sents, gt_sents = translate_batch(model, model_cfg, iterator, tokenizer_en, src_field, trg_field, device, src_lang, trg_lang, max_len=max_len)
  pred_sents = [pred_sent for pred_sent in pred_sents]
  gt_sents = [[gt_sent] for gt_sent in gt_sents]
  score = bleu_score(pred_sents, gt_sents)
  print(f'BLEU score = {score*100:.3f}')
  return score

