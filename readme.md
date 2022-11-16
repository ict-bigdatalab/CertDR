# CertDR

Repo for our paper, Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models, by Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yixing Fan, Maarten De Rijke, and Xueqi Cheng.
# Citation
If you find our work useful, please consider citing our paper:
```
@inproceedings{wu2022certified,
  title={Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models},
  author={Wu, Chen and Zhang, Ruqing and Guo, Jiafeng and Chen, Wei and Fan, Yixing and de Rijke, Maarten and Cheng, Xueqi},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={2128--2137},
  year={2022}
}
```
# A brief intro of how to use

1. data process
 - generate randomized smoothed tables ->  data_process.py
 - generate perturbed training documents -> perturb_training_docs/generate_noised_docs.py -> perturb_training_docs/convert_perutrbed_train_text_to_tokenized.py -> perturb_training_docs/convert_collection_to_memmap.py
2. noisy_train the model  ->  noisy_train.py
3. test process
 - generate randomized smoothed test docs ->  test_process/get_randomized_testdataset.py
 - evaluate the model over these test docs -> test_process/evalate_fRS.py
4. get the CRQ value -> evaluate/crq_evaluate.py

We thank [Mao Ye](https://lushleaf.github.io/) for his help with this work.
# License
This project is under Apache License 2.0.
