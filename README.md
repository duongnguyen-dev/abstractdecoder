# :open_book: :robot: AbstractDECODER 
<p align="center">
  <img src="https://github.com/duongnguyen-dev/abstractdecoder/blob/main/assets/abstract-decoder-image.png" width="30%" height="30%" />
</p>

- An NLP-powered tool designed to classify each sentence of a clinical trial abstract into its specific role.

## Project Description
AbstractDecoder is an NLP-powered tool designed to automate the classification and summarization of clinical trial abstracts using the PubMed RCT dataset. With the exponential growth of biomedical research, AbstractDecoder aims to simplify the process of extracting key information from randomized controlled trials (RCTs) by identifying crucial elements such as objectives, methods, results, and conclusions. This solution helps researchers, healthcare providers, and policymakers quickly grasp the essential insights from studies, enhancing decision-making and knowledge dissemination.

## Todo
- [x] Training pipeline on CPU
- [ ] Support training on GPU
- [ ] UI for usecase
- [ ] Everthing you need to serve this model on Cloud

## My model 
<p align="center">
  <img src="https://github.com/duongnguyen-dev/abstractdecoder/blob/main/assets/abstractdecoder.drawio.png" width="50%" height="50%" />
</p>

## Comparing with previous model

## Environment setup
- Requirement: Python 3.9, [Conda](https://docs.anaconda.com/miniconda/)
- Clone this repositor:<br>
```
git clone --recurse-submodules https://github.com/duongnguyen-dev/abstractdecoder.git
```

- Create conda environment and related packages:<br>
```
conda create -n abstractdecoder python=3.9
conda activate abstractdecoder
pip install -e .
```
## References
- This project uses dataset from [PubMed RCT Dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
- My work is trying to surpass the model performance from this paper [Neural Networks for Joint Sentence Classification
in Medical Paper Abstracts](https://arxiv.org/pdf/1612.05251)
