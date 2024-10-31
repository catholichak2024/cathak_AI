import pandas as pd
from rouge_score import rouge_scorer
import nltk.translate.bleu_score as bleu
from bert_score import score
from reco_metric import start_point, go
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False



df = pd.read_excel('리코평가_사람.xlsx')
df = pd.DataFrame(df)


references = []
for i in df['답변']:
    references.append(i)


candidate = '''"물론이죠! 예술 관련 교양과목 중 추천할 만한 과목 몇 가지가 있습니다. 먼저 '현대 미술의 이해' 과목은 추천 별점이 4.5점이며, 평가가 중간 시험과 기말 레포트로 이루어져 있어 부담 없이 예술의 흐름을 이해할 수 있어요. 이 과목은 중핵 교양으로 학문적 기초를 다지기에 좋습니다."'''
candidate1 = ''' '컴퓨터 구조' 과목은 추천 별점이 4.6점으로 높고, 시험이 없이 과제와 발표 위주로 진행되어 있습니다. 이 과목은 컴퓨터의 구조와 이론를 다루며, 컴퓨터의 작동원리를 학습할 수 있는 중핵 교양과목입니다.'''
candidate2 = '''"알겠습니다! 졸업 요건을 위해 '사회와 법' 과목을 추천드려요. 이 과목은 필수 교양에 속하며 추천 별점이 4.7점으로 학생들에게. 시험은 중간과 기말로 나누어져 있지만, 과목 내용이 실생활과 밀접해 있어 흥미롭게 들을 수 있을 거예요."'''

print('BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate.split()))
print('BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate1.split()))
print('BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate2.split()))


a = []
b = []
c = []
for i in range(3):
    # ROUGE 점수 초기화
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Scorer 생성
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 각 참조 문장에 대해 ROUGE 점수 계산
    for ref in references:
        scores = scorer.score(ref, candidate)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    a.append(average_rouge1)
    b.append(average_rouge2)
    c.append(average_rougeL)

print("ROUGE-1 평균:", sum(a)/3)
print("ROUGE-2 평균:", sum(b)/3)
print("ROUGE-L 평균:", sum(c)/3)



P, R, F1 = score([candidate]*105, references, lang="kr", verbose=True)

print("BERTScore Precision:", P.mean().item())
print("BERTScore Recall:", R.mean().item())
print("BERTScore F1 Score:", F1.mean().item())



plt.figure(figsize=(10, 6))

# 사람 평가에 대한 히스토그램 생성
sns.histplot(df['G-eval'], bins=5)  # bins를 5로 설정하여 1-5 범위로 나누기
plt.xlim(1, 5)  # x축의 범위를 1에서 5로 설정
plt.xticks(range(1, 6))  # x축의 눈금을 1, 2, 3, 4, 5로 설정
plt.title("G-eval 분포 히스토그램")
plt.xlabel("G-eval (1-5)")
plt.ylabel("갯수")
plt.grid(axis='y')  # y축에 그리드 추가
plt.show()

# 사람 평가 점수 분포
score_counts = df['G-eval'].value_counts().sort_index()
# 파이 차트 생성
plt.figure(figsize=(8, 8))
plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("G-eval 점수 분포")
plt.axis('equal')  # 원형 차트를 유지하기 위해 x, y 비율을 같게 설정
plt.show()
