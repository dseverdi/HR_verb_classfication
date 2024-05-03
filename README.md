# Croatian Verb Inflection Model

This repository contains the implementation of a neural computation model for inflecting Croatian verbs. The model can map verbs from their infinitive form to their present form  and vice versa using a convolutional neural network architecture.

## Abstract

All verbal forms in the Croatian language can be derived from two basic forms: the infinitive and the present stems. In this paper, we present a neural computation model that takes a verb in an infinitive form and finds a mapping to a present form. The same model can be applied vice-versa, i.e., map a verb from its present form to its infinitive form. Knowing the present form of a given verb, one can deduce its inflections using grammatical rules. We experiment with our model on the Croatian language, which belongs to the Slavic group of languages. The model learns a classifier through these two classification tasks and uses class activation mapping to find characters in verbs contributing to classification. The model detects patterns that follow established grammatical rules for deriving the present stem form from the infinitive stem form and vice-versa. If mappings can be found between such slots, the rest of the slots can be deduced using a rule-based system.

## Model Description

We propose a neural-network-based computation model that learns to map Croatian verbs from the infinitive stem form to the present stem form and vice-versa. It is considered a classification problem. Our model is essentially a convolutional neural network, and  empirically examines the appropriateness of such architecture.



## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/croatian-verb-inflection-model.git
```


3. Run the experiments in Jupyter notebooks `CNN.ipynb` or `sigmorphon.ipynb`



## Contributors

- [Domagoj Ševerdija](https://github.com/dseverdi)
- [Rebeka Čorić](https://scholar.google.hr/citations?user=981o0gMAAAAJ&hl=hr)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research was supported by ADRIS Zaklada grant 2019.
- We thank Mario Essert for his valuable feedback and contributions.
