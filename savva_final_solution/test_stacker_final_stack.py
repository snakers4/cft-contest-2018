
# coding: utf-8


from local_utils import *

model_name = "stack_final"


print("## Test stacking")



test_df = pd.read_parquet(data_dir+"test_df.parquet", columns=["id", "fullname"])
print(test_df.shape)


# Savva's models
select_cols = ["id", "prob", "pred", "fullname_pred"]
ssa_df = pd.read_parquet(results_dir+"test_spell_simple_seq2seq.parquet")[select_cols].sort_values("id").reset_index(drop=True)
ssa1_df = pd.read_parquet(results_dir+"test_spell_simple_seq2seq1.parquet")[select_cols].sort_values("id").reset_index(drop=True)
ssea_df = pd.read_parquet(results_dir+"test_spell_simple_emb_seq2seq.parquet")[select_cols].sort_values("id").reset_index(drop=True)


# Alex's models   
l4_df = load_l4_df("pred_encdec_savva_l4_h512e512_1e4.csv")
l4_aug2_df = load_l4_df("pred_encdec_savva_l4_h512e512_1e4_aug2.csv")
l4_heavy_resume_df = load_l4_df("pred_encdec_savva_l4_heavy_h512e512_1e4_aug3_lrf01_resume.csv")
l4_aug2_pseudo_06_df = load_l4_df("pred_aug2_pseudo_06.csv")
l4_heavy_dec_skip_final_df = load_l4_df("pred_heavy_dec_skip_final.csv")
l4_aug2_pseudo_06_final_df = load_l4_df("pred_aug2_pseudo_06_final.csv")
l4_aug2_pseudo_10_final2_df = load_l4_df("pred_aug2_pseudo_10_final2.csv")
l4_aug2_pseudo5x_final2_df = load_l4_df("pred_aug2_pseudo5x_final2.csv")
l4_train_aug_correct_only_final2_df = load_l4_df("pred_train_aug_correct_only_final2.csv")


pred_df = test_df.copy()
pred_df["pred"] = np.argmax(np.stack(
                                     ssa_df.prob.values*\
                                     ssa1_df.prob.values*\
                                     ssa1_df.prob.values*\
                                     ssea_df.prob.values*\
                                     l4_df.prob.values*\
                                     l4_heavy_resume_df.prob.values*\
                                     l4_train_aug_correct_only_final2_df.prob.values*\
                                     l4_train_aug_correct_only_final2_df.prob.values*\
                                     l4_train_aug_correct_only_final2_df.prob.values
                                    ), axis=1)

pred_df["fullname_pred"] = l4_train_aug_correct_only_final2_df.fullname_pred.values

pred_df["match"] = (pred_df.fullname == pred_df.fullname_pred).astype(int)
fix_cond = (pred_df.pred == 1) & (pred_df.match == 1)
pred_df.loc[fix_cond, "fullname_pred"] = ssa1_df[fix_cond].fullname_pred
pred_df.loc[fix_cond, "pred"] = l4_train_aug_correct_only_final2_df[fix_cond].pred

pred_df["match"] = (pred_df.fullname == pred_df.fullname_pred).astype(int)
pred_df.loc[(pred_df.pred == 0) & (pred_df.match == 0), "pred"] = 1


print("## Tokens voting")



def split_pad(name, max_len=10):
    name_tokens = name.split()[:max_len]
    name_tokens += [''] * (max_len - len(name_tokens))
    return np.asarray(name_tokens)

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def name_tokens_voting(name_stack):
    name_tokens_stack = np.stack([split_pad(name) for name in name_stack], axis=1)
    return " ".join([most_common(token) for token in name_tokens_stack]).strip()

fullname_stack = list(zip(
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,
    l4_train_aug_correct_only_final2_df["fullname_pred"].values,

    l4_aug2_pseudo5x_final2_df["fullname_pred"].values,
    l4_aug2_pseudo5x_final2_df["fullname_pred"].values,
    l4_aug2_pseudo5x_final2_df["fullname_pred"].values,
    
    l4_aug2_pseudo_10_final2_df["fullname_pred"].values,
    l4_aug2_pseudo_10_final2_df["fullname_pred"].values,

    l4_aug2_pseudo_06_final_df["fullname_pred"].values,
    l4_heavy_dec_skip_final_df["fullname_pred"].values,
    l4_aug2_pseudo_06_df["fullname_pred"].values,
    l4_heavy_resume_df["fullname_pred"].values,
    l4_aug2_df["fullname_pred"].values,
    l4_df["fullname_pred"].values,
    ssa1_df["fullname_pred"].values,
    ssa_df["fullname_pred"].values,
    ssea_df["fullname_pred"].values
))
print(len(fullname_stack[0]))


test_names_pred = np.array(list_multiprocessing(fullname_stack, name_tokens_voting, workers=cpu_cores))


pred_df["fullname_pred"] = test_names_pred.copy()

pred_df["match"] = (pred_df.fullname == pred_df.fullname_pred).astype(int)
fix_cond = (pred_df.pred == 1) & (pred_df.match == 1)
pred_df.loc[fix_cond, "fullname_pred"] = ssa1_df[fix_cond].fullname_pred
pred_df.loc[fix_cond, "pred"] = l4_train_aug_correct_only_final2_df[fix_cond].pred

pred_df["match"] = (pred_df.fullname == pred_df.fullname_pred).astype(int)
pred_df.loc[(pred_df.pred == 0) & (pred_df.match == 0), "pred"] = 1


pred_df.to_parquet(results_dir+"test_"+model_name+"_df.parquet")


print("## Preparing submission")


submission_name = model_name+".csv"

test_df = pred_df
test_df.loc[test_df.pred != 1, "fullname_pred"] = ""
test_df.rename(columns={"pred": "target", "fullname_pred": "fullname_true"})[["id", "target", "fullname_true"]].to_csv(results_dir + submission_name, index=False)
FileLink(results_dir + submission_name)

