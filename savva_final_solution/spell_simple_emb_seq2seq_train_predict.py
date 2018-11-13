
# coding: utf-8



from local_utils import *
model_name = "spell_simple_emb_seq2seq"



print("## Load data")



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
                                   ohe_input=False, batch_size=inf_batch_size)

val_eval_seq = SimplePredGenerator(val_df.fullname.values, convert2seq=chars2seq, 
                                   vocab_size=VOCAB_SIZE, conv_extra=conv_extra,
                                   ohe_input=False, batch_size=inf_batch_size)



print("## Training model")



def buildSpellEmbModel(vocab_size):
    x_input = Input(shape=(None,), name="input_emb_seq")
    embs = Embedding(vocab_size, 256, trainable=True, name='embs')(x_input)
#     embs = Activation("tanh", name="embs_activation")(embs)
    embs = SpatialDropout1D(0.25, name="embs_spdp")(embs)

    bi_rnn1 = Bidirectional(CuDNNLSTM(256, return_sequences=True), name="bi_rnn1")(embs)
    bi_rnn1 = SpatialDropout1D(0.25, name="bi_rnn1_spdp")(bi_rnn1)
    bi_rnn2 = Bidirectional(CuDNNLSTM(256, return_sequences=True), name="bi_rnn2")(bi_rnn1)
    bi_rnn2 = SpatialDropout1D(0.25, name="bi_rnn2_spdp")(bi_rnn2)
    bi_rnn3 = Bidirectional(CuDNNLSTM(256, return_sequences=True), name="bi_rnn3")(bi_rnn2)
    bi_rnn3 = SpatialDropout1D(0.25, name="bi_rnn3_spdp")(bi_rnn3)
    bi_rnn4 = Bidirectional(CuDNNLSTM(256, return_sequences=True), name="bi_rnn4")(bi_rnn3)
    bi_rnn4 = SpatialDropout1D(0.25, name="bi_rnn4_spdp")(bi_rnn4)
    skip_con = concatenate([embs, bi_rnn1, bi_rnn2, bi_rnn3, bi_rnn4], name="skip_connection")
    
    attention, att_weights = AttentionWeightedAverage(name="attention", return_attention=True)(skip_con)
    spell_pred = Dense(3, activation="softmax", name="spell_pred")(attention)
    
    att_weights = Lambda(lambda x: K.expand_dims(x), name="att_weights")(att_weights)
    name_skip_con = concatenate([skip_con, att_weights], name="concat_att_weights")
    name_pred = Conv1D(vocab_size, 5, padding="valid", activation="softmax", name="name_pred")(name_skip_con)
    
    return Model(inputs=x_input, outputs=[spell_pred, name_pred], name="spell_emb_model")

K.clear_session()
model = buildSpellEmbModel(VOCAB_SIZE)
model.summary()



trn_seq = SimpleGenerator(trn_df.fullname.values, 
                          trn_df.fullname_true.values, 
                          trn_df.target.values, 
                          vocab_size=VOCAB_SIZE, convert2seq=chars2seq,
                          ohe_input=False, ohe_output=True,
                          batch_size=batch_size, shuffle=True)

model_checkpoint = ModelCheckpoint(models_dir+model_name+'.h5', monitor='val_score', mode='max', 
                                   verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_score', min_delta=0, patience=12, mode='max', verbose=1)
val_eval = Evaluation(val_eval_seq, val_df.target.values, val_df.fullname_true.values, "val")



epochs=32
clr = CyclicLR(base_lr=0.0003, max_lr=0.003, step_size=4*len(trn_seq), mode='triangular2')
model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
              loss_weights=[1.0, 2.0], metrics=["accuracy"], 
              optimizer=optimizers.Adam(amsgrad=True))

model.fit_generator(
    generator=trn_seq, steps_per_epoch=len(trn_seq),
    initial_epoch=0, epochs=epochs, shuffle=False, verbose=1,
    callbacks=[clr, val_eval, model_checkpoint, early_stop],
    use_multiprocessing=False, workers=cpu_cores, max_queue_size=2*cpu_cores+2)



# more training
model.fit_generator(
    generator=trn_seq, steps_per_epoch=len(trn_seq),
    initial_epoch=epochs, epochs=epochs*2, shuffle=False, verbose=1,
    callbacks=[clr, val_eval, model_checkpoint, early_stop],
    use_multiprocessing=False, workers=cpu_cores, max_queue_size=2*cpu_cores+2)



print("## Model evaluation and predicting")



del model
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
                               ohe_input=False, batch_size=inf_batch_size)



test_spell_prob, test_names_pred = predict(model, test_seq)
test_spell_pred = np.argmax(test_spell_prob, axis=1)

test_df["prob"] = list(test_spell_prob)
test_df["pred"] = test_spell_pred
test_df["fullname_pred"] = test_names_pred
test_df["match"] = (test_df.fullname == test_df.fullname_pred).astype(int)
test_df = test_df.sort_values("id").reset_index(drop=True)

test_df.to_parquet(results_dir+"test_"+model_name+".parquet")

