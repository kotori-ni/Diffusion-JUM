import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=500, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = self.len // self.window
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        num_segments = data.shape[0] // self.window
        x = data.reshape(num_segments, self.window, self.var_num)

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv robustly.
        - 自动检测是否存在 header（如 'sub01,sub02,...'）
        - 删除 pandas 自动产生的 Unnamed 列（通常是保存时带 index 导致）
        - 返回 (numpy array, fitted MinMaxScaler)
        """
        # 先用 header=None 读取，检查首行是否为非数值（即可能为列名）
        try:
            df_try = pd.read_csv(filepath, header=None)
        except Exception as e:
            raise RuntimeError(f"读取 CSV 失败: {filepath}\n{e}")

        # 检查首行是否包含非数值内容（判断为 header 的可能性）
        first_row = df_try.iloc[0].astype(str)
        is_header = True
        for v in first_row:
            # 认为纯数字（含小数/负号）为数值，否则视为 header 字符串
            s = v.strip()
            if s == '':
                # 空字符串也认为非 header（保守）
                is_header = True
                break
            # 尝试把字符串转换为 float 判断
            try:
                float(s)
                # 如果能成功转换为 float，则不是 header -> 继续判断下一个
                # （但遇到若干列均为数值则仍可能没有 header）
                continue
            except:
                # 无法转为 float，则很可能首行为 header（列名）
                is_header = True
                break
        else:
            # 如果循环没有 break（即所有首行都能转成数值），则首行很可能不是 header
            is_header = False

        # 根据检测结果选择读取方式
        if is_header:
            df = pd.read_csv(filepath, header=0)
        else:
            df = pd.read_csv(filepath, header=None)

        # 删除 pandas 可能引入的 unnamed 列（例如保存时带 index 导致的 'Unnamed: 0'）
        unnamed_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Unnamed')]
        if len(unnamed_cols) > 0:
            df = df.drop(columns=unnamed_cols)

        # 有些 CSV 第一列可能不应该存在（比如保存时多余的索引列），如果第一列名或第一列值可疑也删除
        # 进一步检查：如果列名中第一个看起来像索引（例如 '0' 且后续列名很多），可尝试删除它（保守策略）
        if df.shape[1] > 1:
            first_col_vals = df.iloc[:, 0].astype(str)
            # 如果第一列的很多值可解析为整数且随行递增（疑似原索引），则删除
            try:
                as_ints = first_col_vals.str.match(r'^\d+$').sum()
                if as_ints / float(len(first_col_vals)) > 0.9:
                    # 90% 以上为纯整数字符串 -> 很可能是索引列
                    df = df.drop(df.columns[0], axis=1)
            except Exception:
                pass

        # 最后做一次类型转换，确保数据为数值
        df = df.apply(pd.to_numeric, errors='coerce')

        # 如果存在 NaN，给出提示（可能说明 header 没读对）
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            print(f"⚠️ 注意: 读取后发现 {nan_count} 个 NaN，可能是 header/格式问题。文件: {filepath}")

        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        # 可选：打印诊断信息，便于调试
        print(f"[read_data] {os.path.basename(filepath)} -> shape: {data.shape}; columns: {df.shape[1]}")
        return data, scaler

    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
