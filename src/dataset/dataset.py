import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Spam_Dataset(Dataset):

    def __init__(
            self,
            spam_df: pd.DataFrame
    ):
        super().__init__()
        self.df = spam_df
        self.ids = self.df.index.values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return {
            'message': row['message'],
            'is_spam': row['is_spam']
        }


def spam_dataframe(
        path: str,
        is_train: bool = True,
        val_ratio: Optional[float] = 0.1,
        random_state: Optional[int] = 42,
        shuffle: bool = True
):
    df = pd.read_csv(path, encoding='iso-8859-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

    df.columns = ['is_spam', 'message']
    df = df[['message', 'is_spam']]
    df['is_spam'] = df['is_spam'] == 'spam'

    if is_train:
        df = df.dropna(axis=0)
        df.drop_duplicates('message', inplace=True, ignore_index=True)

        if val_ratio:
            train_df, val_df = train_test_split(
                df,
                test_size=val_ratio,
                random_state=random_state,
                shuffle=shuffle
            )
            return train_df, val_df
        else:
            return df
    else:
        if val_ratio:
            print('train/val split has been ignored')
        return df

