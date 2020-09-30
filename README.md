UMPR
===
> Codes for "Recommendation by Users' Multi-modal Preferences for Smart City Applications".  

# Current work

Review Network: Finished  
Visual Network: Not finished  
Control Network: Not finished  

# Preparation

1. Download pre-trained embedding dictionary "glove.twitter.27B.50d.txt" on https://nlp.stanford.edu/projects/glove/  
2. Training data "reviews.json" contains 10000 samples selected from "yelp_academic_dataset_review.json" on https://www.yelp.com/dataset/documentation/main  

# Requirements

tensorflow 1.15.2

# Running

Please first correct the file path at the top of the "main.py"

```shell script
python main.py
```