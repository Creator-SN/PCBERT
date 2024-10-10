## PCBERT: Parent and Child BERT for Chinese Few-shot NER

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

```
@inproceedings{lai2022pcbert,
  title     = {{PCBERT}: Parent and Child BERT for Chinese Few-shot NER},
  author    = {Lai, Peichao and Ye, Feiyang and Zhang, Lin and Chen, Zhiwei and Fu, Yanggeng and Wu, Yingjie and Wang, Yilei},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  pages     = {2199--2209},
  year      = {2022}
}
```
