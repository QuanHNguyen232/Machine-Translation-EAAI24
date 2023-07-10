# Machine-Translation-EAAI24

Discussion:
* Pivot model has en-fr BLEU 33 (much lower than training separately) because we did not assign the correct weight for the loss (hypothesis). Thus we can try dynamic ensemble loss in the future

NEW TASKS:
* [X] Seq2Seq: sort by src_len and unsort output --> ensure output matches with trg
* [X] Pivot model: ensure it works for $n$ seq2seq models
* [ ] Trian model: ensure outputs from all submodels match w/ target sent

### Table of content
1. Config ([go-there](#config))
1. Best models ([go-there](#best-models))
1. Things in common ([go-there](#things-in-common))
---

## Config

* <details><summary>Vocab size</summary>

    Build on first 64000 sentences of data `EnDeFrItEsPtRo-76k-most5k.pkl` with `min_freq=2`:
    * en: 6964
    * fr: 9703
    * es: 10461
    * it: 10712
    * pt: 10721
    * ro: 11989
</details>

* <details><summary>Model config</summary>

    * Embed_Dim = 256
    * Hidden_Dim = 512
    * Dropout = 0.5
</details>

## Best models:
* Seq2Seq: `seq2seq-EnFr-1.pt`
* Pivot:
    * es: `piv-EnEsFr.pt`
    * it: `piv-EnItFr.pt`
    * pt: `piv-EnPtFr.pt`
    * ro: `piv-EnRoFr.pt`
* Triang: (combination of trained Seq2Seq & Pivot)

## Things in common:
* Main pipeline: `bentrevett_pytorch_seq2seq.ipynb`
* Datasets: `EnDeFrItEsPtRo-76k-most5k.pkl`
* Data info: train_len, valid_len, test_len = 64000, 3200, 6400
* <details><summary>Init weights (all models)</summary>

    ```python
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)    
    model.apply(init_weights);
    ```
    </details>

* <details><summary>Load model weights</summary>

    ```python
    checkpoint = torch.load('path_to_model/model_name.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    ```
    </details>

* <details><summary>Learning rate</summary>

    * Seq2Seq: start w/ $0.0012$, reduced by $\frac{2}{3}$ every epoch
    * Pivot: start w/ $0.0012$, reduced by $\frac{2}{3}$ at epoch 3rd, 6th, 8th, 9th, 10th.
</details>

* <details><summary>Epoch</summary>

    * Seq2Seq: 7
    * Pivot: 11
</details>



<details>
    <summary>Future work improvements for dataloader(after having results)</summary>

* Replace Field, BucketIterator with those:
    1. Use `build_vocab_from_iterator` ([example tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html))
    2. [torchtext.vocab](https://pytorch.org/text/stable/vocab.html)
    3. [torchtext tutorial general](https://pytorch.org/text/0.14.0/)
* `EmbeddingBag` with `offsets` can replace `Embedding` and `sent_len` is OPTIONAL since using `pack_padded_sequence` also reduces padding.

</details>

## Transformer
* Implement based on "Attention is all you need"
    * github most popular: [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master)
    * kaggle implementation by [FERNANDES](https://www.kaggle.com/code/ricafernandes/attention-is-all-you-need-paper-implementation/notebook)
    * another fine github repo [jayparks/transformer](https://github.com/jayparks/transformer/tree/master)
    * a basic explanation of some concepts on [towardsdatascience](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)
    * check VietAI code