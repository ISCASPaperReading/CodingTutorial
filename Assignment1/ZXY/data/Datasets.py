import os

import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'vali': 1}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * 0.7)
        num_vali = int(len(df_raw) * 0.3)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def get_dataset(root_path, data_path, seq_len, label_len, pred_len):
    Data = MyDataset
    train_data = Data(
        root_path=root_path,
        data_path=data_path,
        flag='train',
        size=[seq_len, label_len, pred_len],
    )
    vali_data = Data(
        root_path=root_path,
        data_path=data_path,
        flag='vali',
        size=[seq_len, label_len, pred_len],
    )
    return train_data, vali_data
