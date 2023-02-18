from src import *
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'dataset/spam-ham v2.csv'
df = spam_dataframe(PATH, True, None)

print(df.info())
print(df.value_counts('is_spam'))

##
# class distribution

sns.countplot(data=df, x='is_spam')
plt.show()

##



##
# sequence length distribution

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader

CHECKPOINTS = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINTS)

dataset = Spam_Dataset(df)
collator = Spam_Collator(tokenizer, None)
loader = DataLoader(dataset, collate_fn=collator, batch_size=1)

##
seq_len = []
for i in loader:
    seq_len.append(i[0]['input_ids'].shape[-1])

seq_len = pd.Series(seq_len)
print(seq_len.describe())

sns.boxplot(seq_len)
plt.show()

##

