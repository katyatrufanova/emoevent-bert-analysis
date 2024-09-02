# EmoEvent BERT Analysis: Advancing Emotion Detection in Multilingual Social Media Data

## Project Overview

This project builds upon the work of Plaza-del-Arco et al. in their paper "EmoEvent: A Multilingual Emotion Corpus based on different Events". It extends their research by implementing and evaluating state-of-the-art deep learning models for emotion classification in social media data.

The primary focus of this project is to explore the effectiveness of BERT (Bidirectional Encoder Representations from Transformers) in detecting emotions in tweets, addressing one of the key suggestions for future work proposed by the original authors.

## Dataset

This project utilizes the EmoEvent dataset, a multilingual corpus of tweets annotated for emotions and offensiveness. The dataset is based on eight significant events from April 2019 and is available in both English and Spanish. The analysis presented in this project focuses on the English portion of the dataset.

Dataset Source: [EmoEvent Dataset on Hugging Face](https://huggingface.co/datasets/fmplaza/EmoEvent)

## Methodology

Building upon the original research, this project implements a series of experiments using BERT, a powerful transformer-based model known for its effectiveness in various NLP tasks. The experiments were designed to explore different hyperparameter configurations and training strategies:

1. **Experiment 1 (BERT 3e)**: 
   - Optimizer: Adam
   - Learning Rate: 5e-5
   - Epochs: 3
   - Batch Size: 32
   - Loss Function: Sparse Categorical Cross Entropy

2. **Experiment 2 (BERT 10e)**:
   - Same configuration as Experiment 1, but trained for 10 epochs to explore the impact of extended training.

3. **Experiment 3 (BERT Weighted)**:
   - Optimizer: Adam
   - Learning Rate: 3e-5 (reduced)
   - Epochs: 5
   - Batch Size: 16 (reduced)
   - Additional Features: Class weights and early stopping

These experiments were designed to systematically explore the impact of various hyperparameters on the model's performance in emotion classification.

## Results and Analysis

The experiments yielded insightful results:

![results](https://github.com/user-attachments/assets/d3fb5b3f-2d8c-446b-b9da-1f0768f5b3ba)

1. **BERT 3e** demonstrated the best overall performance, achieving the highest scores across accuracy, precision, recall, and F1-score. This model showed a strong ability to correctly classify tweets into emotional categories while maintaining a good balance between precision and recall.

2. **BERT 10e** and **BERT Weighted** showed good precision but slightly lower recall compared to BERT 3e. This suggests that while these models were accurate in their predictions, they might have underrepresented some emotional categories.

3. All BERT models significantly outperformed the baseline SVM model used in the original paper, highlighting the potential of deep learning approaches in emotion classification tasks.

These results underscore the effectiveness of BERT in capturing the nuanced emotional content of social media data, while also demonstrating the importance of careful hyperparameter tuning in maximizing model performance.

## Project Organization
```
emoevent-bert-analysis/
├── docs/
│ ├── Presentation.pdf
│
├── notebooks/
│ ├── bert_3e_experiment.ipynb
│ ├── bert_10e_experiment.ipynb
│ └── bert_weighted_experiment.ipynb
│
├── results/
│ └── performance_comparison.csv
│
└── README.md
```

## Getting Started

1. Clone this repository:
```
git clone https://github.com/katyatrufanova/EmoEvent-BERT-Analysis.git
```

2. Upload the [EmoEvent Dataset](https://huggingface.co/datasets/fmplaza/EmoEvent) to your Google Drive in the path: `MyDrive/Sentiment Analysis/`

3. Open the notebooks in Google Colab and run them sequentially.

## Future Work

This project opens up several avenues for future research:

1. Extending the analysis to the Spanish portion of the dataset for a comprehensive multilingual comparison.
2. Exploring other state-of-the-art transformer models like RoBERTa or XLNet for emotion classification.
3. Implementing more sophisticated techniques for handling class imbalance and fine-tuning.
4. Investigating the model's performance on specific event types within the dataset.

## Conclusion

This project demonstrates the potential of advanced deep learning techniques in emotion classification tasks, particularly in the context of social media data. By building upon the foundational work of Plaza-del-Arco et al., we've shown how BERT can be effectively applied to enhance emotion detection accuracy. The systematic exploration of hyperparameters and training strategies provides valuable insights for future research in this domain.

## Acknowledgments

- Arco, F.M., Strapparava, C., López, L.A., & Valdivia, M.T. (2020). EmoEvent: A Multilingual Emotion Corpus based on different Events. *International Conference on Language Resources and Evaluation*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
