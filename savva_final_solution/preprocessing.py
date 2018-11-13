
# coding: utf-8


from processing_utils import *


train_df = pd.read_csv("data/cft-train.zip")
print(train_df.shape)

test_df = pd.read_csv("data/cft-test.zip")
print(test_df.shape)


train_df.loc[train_df["id"] == 286958, "fullname"] = "Ш Р"
train_df.fullname = train_df.fullname.astype(str).apply(cleaner)
test_df.fullname = test_df.fullname.astype(str).apply(cleaner)


train_df.loc[train_df.target != 1, "fullname_true"] = train_df.loc[train_df.target != 1, "fullname"]

train_df["fullname_len"] = train_df.fullname.apply(len)
train_df["fullname_true_len"] = train_df.fullname_true.apply(len)
test_df["fullname_len"] = test_df.fullname.apply(len)


print_len_stats(train_df.fullname_len.values)
print_len_stats(test_df.fullname_len.values)


trn_ids, val_ids = train_test_split(train_df.id.values, stratify=train_df.target.values, test_size=0.1, random_state=seed)
print("train_size: {}, valid_size: {}".format(len(trn_ids), len(val_ids)))


trn_df = train_df[train_df["id"].isin(set(trn_ids))].reset_index(drop=True)
print(trn_df.shape)

val_df = train_df[train_df["id"].isin(set(val_ids))].reset_index(drop=True)
print(val_df.shape)


trn_df["fullname_seq"] = trn_df.fullname.apply(chars2seq)
trn_df["fullname_true_seq"] = trn_df.fullname_true.apply(chars2seq)

val_df["fullname_seq"] = val_df.fullname.apply(chars2seq)
val_df["fullname_true_seq"] = val_df.fullname_true.apply(chars2seq)

test_df["fullname_seq"] = test_df.fullname.apply(chars2seq)


trn_df.to_parquet("data/trn_df.parquet")
val_df.to_parquet("data/val_df.parquet")
test_df.to_parquet("data/test_df.parquet")
