## PCBERT: Parent and Child BERT for Chinese Few-shot NER
---

Achieving good performance on few-shot or zero-shot datasets has been a long-term challenge for NER. The conventional semantic transfer approaches on NER will decrease model performance when the semantic distribution is quite different, especially in Chinese few-shot NER. Recently, prompt-tuning has been thoroughly considered for low-resource tasks. But there is no effective prompt-tuning approach for Chinese few-shot NER. In this work, we propose a prompt-based Parent and Child BERT (PCBERT) for Chinese few-shot NER. To train an annotating model on high-resource datasets and then discover more implicit labels on low-resource datasets. We further design a label extension strategy to achieve label transferring from high-resource datasets. We evaluated our model on Weibo and the other three sampling Chinese NER datasets, and the experimental result demonstrates our approach's effectiveness in few-shot learning.

## Overview
---

![image](https://github.com/user-attachments/assets/579ffcfa-aa57-4762-b995-db3701663c18)

## Run

Train Steps:

1. Use LEBERT train from SUPERNER datasets.
2. Predict target domain train dataset.
3. Use predict dataset train P-BERT.
4. Load P-BERT and fine-tune C-BERT.

Please check out the `exp.ipynb` to run the project.

## Dataset

The sampled datasets of low-resource scenarios in this work include: `Weibo`, `Ontonotes`, `MSRA`, and `Resume`.

You can download the proceed low-resource and high-resource datasets in [Google Drive](https://drive.google.com/file/d/1TH6s_njTI04f2YZWc4S3WL4IifRYy43-/view?usp=sharing).

## Citation

```bib
@inproceedings{lai2022pcbert,
  title     = {{PCBERT}: Parent and Child BERT for Chinese Few-shot NER},
  author    = {Lai, Peichao and Ye, Feiyang and Zhang, Lin and Chen, Zhiwei and Fu, Yanggeng and Wu, Yingjie and Wang, Yilei},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  pages     = {2199--2209},
  year      = {2022}
}
```
