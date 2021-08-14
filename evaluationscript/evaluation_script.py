import nltk
import sys
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score

file1 = open(sys.argv[1], 'r')
references = file1.readlines()
file2 = open(sys.argv[2], 'r')
hypotheses = file2.readlines()

total_num = len(references)
total_bleu_scores = 0
total_meteor_scores = 0
for i in range(total_num):
  total_bleu_scores+=sentence_bleu([references[i].split(" ")], hypotheses[i].split(" "))
  total_meteor_scores+=single_meteor_score(references[i], hypotheses[i])

bleu_result = total_bleu_scores/total_num
meteor_result = total_meteor_scores/total_num

print("bleu score: ",bleu_result)
print("meteor score: ",meteor_result)