# Machine-Translation-EAAI24

### Table of content
1. Best models ([go-there](#best-models))
1. Things in common ([go-there](#things-in-common))
---

## Best models:
* Seq2Seq:
    * en-fr: `attn_enfr_160kset.pt` (BLEU = 32.18)
        <details>
        <summary>Model detail</summary>

        ```python
        # Check node-data.md to get FIELD.vocab
        INPUT_DIM = 2463    #len(SRC_FIELD.vocab)
        OUTPUT_DIM = 2495   #len(TRG_FIELD.vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        DEC_HID_DIM = 512
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        LR = 0.001
        SRC_PAD_IDX = SRC_FIELD.vocab.stoi[SRC_FIELD.pad_token]

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

        model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.333)
        ```
        </details>
* Pivot:
    * en-de-fr: `piv_endefr_74kset_2.pt` (BLEU = 26.33)
        <details>
        <summary>Model detail</summary>

        ```python
        # Check node-data.md to get FIELD.vocab
        INPUT_DIM = 2267    #len(SRC_FIELD.vocab)
        PIV_DIM = 2474    #len(PIV_FIELD.vocab)
        OUTPUT_DIM = 2390   #len(TRG_FIELD.vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        DEC_HID_DIM = 512
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        LR = 0.001

        SRC_PAD_IDX = SRC_FIELD.vocab.stoi[SRC_FIELD.pad_token]
        PIV_PAD_IDX = PIV_FIELD.vocab.stoi[PIV_FIELD.pad_token]

        attn1 = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec1 = Decoder(PIV_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn1)
        model1 = Seq2Seq(enc1, dec1, SRC_PAD_IDX, device).to(device)

        attn2 = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc2 = Encoder(PIV_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec2 = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn2)
        model2 = Seq2Seq(enc2, dec2, PIV_PAD_IDX, device).to(device)

        model = PivotSeq2Seq(model1, model2, SRC_FIELD, PIV_FIELD, TRG_FIELD, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.333)

        criterion1 = nn.CrossEntropyLoss(ignore_index = PIV_FIELD.vocab.stoi[PIV_FIELD.pad_token])
        criterion2 = nn.CrossEntropyLoss(ignore_index = TRG_FIELD.vocab.stoi[TRG_FIELD.pad_token])
        criterions = (criterion1, criterion2)
        ```
        </details>

## Things in common:
* Main pipeline: `bentrevett_pytorch_seq2seq.ipynb`
* Datasets:
    * Seq2Seq:
        * en-fr: `endefr_75kpairs_2k5-freq-words.pkl`
    * Pivot:
        * en-de-fr: `enfr_160kpairs_2k5-freq-words.pkl`
* Data info:
    * train_len, valid_len, test_len = 64000, 3200, 6400
    * batch_size = 64

* Init weights: all models use:
    <details>
    <summary>Code</summary>

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
* Load model weights:
    <details>
    <summary>Code</summary>

    ```python
    checkpoint = torch.load('path_to_model/model_name.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    ```
    </details>
