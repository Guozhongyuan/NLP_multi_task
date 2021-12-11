# PRETRAIN MODEL
Need to convert and merge the downloaded model (two file) into one. Use convert.py


# TASK

## dialog
Just learn to generate

## fill in the blank
Each sentence has 10 choices. Distill model example: output-of-GPT2 [1,666,30000] -> scores [1,30000] -> scores-in-choices [1,10] -> argmax to choose the best choice

## classification(zero shot learning)
No need to train. For one news-title, pad 15 classes to generate 15 sentences, and then calculate losses, use 'argmin' to select the class which make the news-title has min loss.  
Result: 30% accuracy on distill model, 43% accuracy on large model.
# NLP_multi_task_singleGPU
