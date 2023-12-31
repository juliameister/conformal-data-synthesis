# Confident data synthesis

This repository is in support of our paper "Conformalised data synthesis with statistical quality guarantees", submitted to the Machine Learning Journal. We show both the code used for results generation, and the specific results presented in the paper. Our preprint describes the method and results interpretation in more depth: https://arxiv.org/abs/2312.08999.

With the proliferation of ever more complicated Deep Learning architectures, data synthesis is a highly promising technique to address the demand of data-hungry models. However, reliably assessing the quality of a `synthesiser' model's output is an open research question with significant associated risks for high-stake domains. To address this challenge, we have designed a unique confident data synthesis algorithm that introduces statistical confidence guarantees through a novel extension of the Conformal Prediction framework. We support our proposed algorithm with theoretical proofs and an extensive empirical evaluation of five benchmark datasets. To show our approach's versatility on ubiquitous real-world challenges, the datasets were carefully selected for their variety of difficult characteristics: low sample count, class imbalance and non-separability, and privacy-sensitive data. In all trials, training sets extended with our confident synthesised data performed at least as well as the original, and frequently significantly improved Deep Learning performance by up to +65% F1-score.

---

## Citation
Meister, J. A., Nguyen, K. A. (2023). Conformalised data synthesis with statistical quality guarantees. ArXiv. https://doi.org/10.48550/arXiv.2312.08999.


@misc{meister2023conformalised,
    title={Conformalised data synthesis with statistical quality guarantees}, 
    author={Julia A. Meister and Khuong An Nguyen},
    year={2023},
    eprint={2312.08999},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    doi={10.48550/arXiv.2312.08999}
}