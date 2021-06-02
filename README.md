  
   ### Speech Function Classifier
## Description
Speech Function classifier is created for analysis of dialogue structure (based on the theory by Eggins&Slade). Overall,there are 32 possible labels. One label is assigned to one utterance. 

## How to use speech function classifier
At first, preprocess a dialogue for analysing its scructure. Get two lists with all utterances and a sequence of speakers. 
```python
import sf_classifier
sf_classifier.classify_labels(utterances,speakers)
```
You will get a list with predicted speech functions as a result.
