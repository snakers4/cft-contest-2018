
# coding: utf-8


from local_utils import *

model_name = "spell_simple_seq2seq1"



print("## Loading data")



trn_df = pd.read_parquet(data_dir+"trn_df.parquet").drop(columns=["fullname_seq", "fullname_true_seq"])
print(trn_df.shape)

val_df = pd.read_parquet(data_dir+"val_df.parquet").drop(columns=["fullname_seq", "fullname_true_seq"])
print(val_df.shape)



conv_extra = 2
batch_size = 512
inf_batch_size = 1024

def chars2seq(text):
    return conv_extra*[CHAR2INX[STR_TOK]] + [CHAR2INX[c] for c in list(text)] + conv_extra*[CHAR2INX[END_TOK]]

trn_eval_df = trn_df.sort_values("fullname_true_len")[["id", "fullname", "fullname_true", "target"]].reset_index(drop=True).copy()

val_df = val_df.sort_values("fullname_true_len").reset_index(drop=True)



trn_eval_seq = SimplePredGenerator(trn_eval_df.fullname.values, convert2seq=chars2seq, 
                                   vocab_size=VOCAB_SIZE, conv_extra=conv_extra,
                                   ohe_input=True, batch_size=inf_batch_size)

val_eval_seq = SimplePredGenerator(val_df.fullname.values, convert2seq=chars2seq, 
                                   vocab_size=VOCAB_SIZE, conv_extra=conv_extra,
                                   ohe_input=True, batch_size=inf_batch_size)


print("## Model evaluation and predicting")



show_cols = ["id", "fullname", "fullname_true", "fullname_pred", "target", "pred", "match", "match_true"]



K.clear_session()
model = keras.models.load_model(models_dir+model_name+'.h5', compile=False, 
                                custom_objects={'AttentionWeightedAverage':AttentionWeightedAverage})



trn_spell_prob, trn_names_pred = predict(model, trn_eval_seq)
trn_spell_pred = np.argmax(trn_spell_prob, axis=1)

trn_task1_score = score_task1(trn_eval_df.target.values, trn_spell_pred)
trn_task2_score = score_task2(trn_eval_df.fullname_true.values, trn_names_pred, 
                              trn_eval_df.target.values, trn_spell_pred)
trn_score = (trn_task1_score+trn_task2_score)/2

print("trn_task1_score: {:.5f}".format(trn_task1_score))
print("trn_task2_score: {:.5f}".format(trn_task2_score))
print("trn_final_score: {:.5f}".format(trn_score))



val_spell_prob, val_names_pred = predict(model, val_eval_seq)
val_spell_pred = np.argmax(val_spell_prob, axis=1)

val_df["prob"] = list(val_spell_prob)
val_df["pred"] = val_spell_pred
val_df["fullname_pred"] = val_names_pred
val_df["match"] = (val_df.fullname == val_df.fullname_pred).astype(int)
val_df["match_true"] = (val_df.fullname_true == val_df.fullname_pred).astype(int)
val_df.sort_values("id").reset_index(drop=True).to_parquet(results_dir+"val_"+model_name+".parquet")

val_task1_score = score_task1(val_df.target.values, val_df.pred.values)
val_task2_score = score_task2(val_df.fullname_true, val_df.fullname_pred, val_df.target.values, val_df.pred.values)
val_score = (val_task1_score+val_task2_score)/2

print("val_task1_score: {:.5f}".format(val_task1_score))
print("val_task2_score: {:.5f}".format(val_task2_score))
print("val_final_score: {:.5f}".format(val_score))



# postprocessing
val_df.loc[(val_df.pred == 0) & (val_df.match == 0), "pred"] = 1

val_task1_score = score_task1(val_df.target.values, val_df.pred.values)
val_task2_score = score_task2(val_df.fullname_true, val_df.fullname_pred, val_df.target.values, val_df.pred.values)
val_score = (val_task1_score+val_task2_score)/2

print("val_task1_score: {:.5f}".format(val_task1_score))
print("val_task2_score: {:.5f}".format(val_task2_score))
print("val_final_score: {:.5f}".format(val_score))



print("## Predicting test")



test_df = pd.read_parquet(data_dir+"test_df.parquet").drop(columns=["fullname_seq"])
print(test_df.shape)



del model
K.clear_session()
model = keras.models.load_model(models_dir+model_name+'.h5', compile=False, 
                                custom_objects={'AttentionWeightedAverage':AttentionWeightedAverage})



test_df = test_df.sort_values("fullname_len").reset_index(drop=True)
test_seq = SimplePredGenerator(test_df.fullname.values, convert2seq=chars2seq, 
                               vocab_size=VOCAB_SIZE, conv_extra=conv_extra,
                               ohe_input=True, batch_size=inf_batch_size)



test_spell_prob, test_names_pred = predict(model, test_seq)
test_spell_pred = np.argmax(test_spell_prob, axis=1)

test_df["prob"] = list(test_spell_prob)
test_df["pred"] = test_spell_pred
test_df["fullname_pred"] = test_names_pred
test_df["match"] = (test_df.fullname == test_df.fullname_pred).astype(int)
test_df = test_df.sort_values("id").reset_index(drop=True)

test_df.to_parquet(results_dir+"test_"+model_name+".parquet")

