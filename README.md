# Virtual Teaching Assistant Pedagogical Evaluation

This is the codebase for our EMNLP 2025 paper.

:warning: $$\color{red}\text{This repository is only for documenting how we processed and generated data, but not for reproduction.}$$ Our data is unfortunately not publicly available for FERPA compliance.

## Repository Structure
- `rubrics/`: The pedagogical assessment rubrics for the five pedagogical capability levels.
- `classify_academic_questions.py`: Classification code for dividing discussion forum posts into different post categories.
- `generate_response_with_discussion.py`: Generation code using Llama-3-70B-Instruct to produce simulated VTA responses. We additionally include the mechanism for using post semantic similarity to retrieve similar peer posts, and the corresponding prompting strategy.
- `optimize_classifier_level1.ipynb`: We used the same code for all levels, here we use level1 as an example. This notebook would extract training and validation examples, and then optimize prompts for the pedagogical assessment classifier using DSPy.
- `synthetic_data.ipynb`: Data synthesis code we used to create the training data for our open-weight classifiers.