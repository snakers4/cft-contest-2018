import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def make_model(src_vocab,
               tgt_vocab,
               device=None,
               emb_size=256,
               hidden_size=512,
               num_layers=1,
               dropout=0.1,
               num_classes=3,
               num_cn=0,
               cn_emb_size=0,
               heavy_decoder=False,
               add_input_skip=False):
    "Helper: Construct a model from hyperparameters."

    if heavy_decoder:
        attention = BahdanauAttention(hidden_size, query_size=hidden_size*2)
        if add_input_skip:
            skip_attention = BahdanauAttention(hidden_size,
                                               query_size=hidden_size*2,
                                               key_size=hidden_size)        
    else:
        attention = BahdanauAttention(hidden_size)
        if add_input_skip:
            skip_attention = BahdanauAttention(hidden_size,
                                               key_size=hidden_size)         
    
    if (num_cn+cn_emb_size) == 0:
        if add_input_skip:
            model = EncoderDecoderSkip(
                Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
                Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout,
                        heavy_decoder=heavy_decoder,add_input_skip=True,skip_attention=skip_attention),
                nn.Embedding(src_vocab, emb_size),
                nn.Embedding(tgt_vocab, emb_size),
                Generator(hidden_size, tgt_vocab),
                Classifier(hidden_size, num_classes=num_classes, dropout=dropout))        
        else:
            model = EncoderDecoder(
                Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
                Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout,
                        heavy_decoder=heavy_decoder,add_input_skip=False),
                nn.Embedding(src_vocab, emb_size),
                nn.Embedding(tgt_vocab, emb_size),
                Generator(hidden_size, tgt_vocab),
                Classifier(hidden_size, num_classes=num_classes, dropout=dropout))
    elif emb_size==1:
        model = EncoderDecoderOhe(
            Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
            Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
            Generator(hidden_size, tgt_vocab),
            Classifier(hidden_size, num_classes=num_classes, dropout=dropout),
            src_vocab,
            tgt_vocab,
            device
        )        
    else:
        model = ConditionalEncoderDecoder(
            Encoder(emb_size+cn_emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
            Decoder(emb_size+cn_emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
            nn.Embedding(src_vocab, emb_size),
            nn.Embedding(tgt_vocab, emb_size),
            Generator(hidden_size, tgt_vocab),
            Classifier(hidden_size, num_classes=num_classes, dropout=dropout),
            nn.Embedding(num_cn, cn_emb_size), # share the country embeddings between encoder and decoder
        )
    return model.to(device)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator, classifier):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.classifier = classifier
        
    def forward(self,
                src, trg,
                src_mask, trg_mask,
                src_lengths, trg_lengths,
                cn):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src,
                                                    src_mask,
                                                    src_lengths,
                                                    cn)
        
        clf_logits = self.classifier(encoder_hidden)
        
        return self.decode(encoder_hidden,
                           encoder_final,
                           src_mask,
                           trg,
                           trg_mask,
                           cn=cn),clf_logits
    
    def encode(self,
               src, src_mask, src_lengths,
               cn):
        return self.encoder(self.src_embed(src),
                            src_mask,
                            src_lengths)
    
    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None,
               cn=None):
        return self.decoder(self.trg_embed(trg),
                            encoder_hidden,
                            encoder_final,
                            src_mask,
                            trg_mask,
                            hidden=decoder_hidden,
                            )

class EncoderDecoderSkip(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator, classifier):
        super(EncoderDecoderSkip, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.classifier = classifier
        
    def forward(self,
                src, trg,
                src_mask, trg_mask,
                src_lengths, trg_lengths,
                cn):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src,
                                                    src_mask,
                                                    src_lengths,
                                                    cn)
        
        clf_logits = self.classifier(encoder_hidden)
        
        return self.decode(encoder_hidden,
                           encoder_final,
                           src_mask,
                           trg,
                           trg_mask,
                           cn=cn,
                           skip=src),clf_logits
    
    def encode(self,
               src, src_mask, src_lengths,
               cn):
        return self.encoder(self.src_embed(src),
                            src_mask,
                            src_lengths)
    
    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None,
               cn=None,
               skip=None):
        return self.decoder(self.trg_embed(trg),
                            encoder_hidden,
                            encoder_final,
                            src_mask,
                            trg_mask,
                            hidden=decoder_hidden,
                            skip=self.src_embed(skip)
                            )    
    
class EncoderDecoderOhe(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator, classifier,
                 src_vocab, tgt_vocab, device):
        super(EncoderDecoderOhe, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.classifier = classifier
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def forward(self,
                src, trg,
                src_mask, trg_mask,
                src_lengths, trg_lengths,
                cn):
        """Take in and process masked src and target sequences."""
        # oh-encode (batch,long index) to (batch,sequence,1)
        src_oh = torch.FloatTensor(src.size(0), src.size(1), self.src_vocab).zero_().to(self.device)
        src_oh.scatter_(2, src, 1)
        
        encoder_hidden, encoder_final = self.encode(src_oh,
                                                    src_mask,
                                                    src_lengths,
                                                    cn)
        clf_logits = self.classifier(encoder_hidden)
        
        return self.decode(encoder_hidden,
                           encoder_final,
                           src_mask,
                           trg,
                           trg_mask,
                           cn=cn),clf_logits
    
    def encode(self,
               src, src_mask, src_lengths,
               cn):
        src_oh = torch.FloatTensor(src.size(0), src.size(1), self.src_vocab).zero_().to(self.device)
        src_oh.scatter_(2, src, 1)        
        return self.encoder(src_oh,
                            src_mask,
                            src_lengths)
    
    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None,
               cn=None):
        trg_oh = torch.FloatTensor(src.size(0), src.size(1), self.tgt_vocab).zero_().to(self.device)
        trg_oh.scatter_(2, trg, 1)           
        return self.decoder(trg_oh,
                            encoder_hidden,
                            encoder_final,
                            src_mask,
                            trg_mask,
                            hidden=decoder_hidden,
                            )    

class ConditionalEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator, classifier, cn_embed):
        super(ConditionalEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.classifier = classifier
        self.cn_embed = cn_embed
        
    def forward(self,
                src, trg,
                src_mask, trg_mask,
                src_lengths, trg_lengths,
                cn):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src,
                                                    src_mask,
                                                    src_lengths,
                                                    cn)
        
        clf_logits = self.classifier(encoder_hidden)
        
        return self.decode(encoder_hidden,
                           encoder_final,
                           src_mask,
                           trg,
                           trg_mask,
                           cn=cn),clf_logits
    
    def encode(self,
               src, src_mask, src_lengths,
               cn):
        
        embedded = self.src_embed(src) # concatenate country embeddings with token embeddings
        
        batch_size = embedded.size(0)
        sequence_size = embedded.size(1)
        cn_embedded = self.cn_embed(cn)
        cn_embed_size = cn_embedded.size(1)
        assert batch_size == cn_embedded.size(0)
        cn_embedded = cn_embedded.unsqueeze(1).expand(batch_size,sequence_size,cn_embed_size).contiguous()
        
        return self.encoder(torch.cat((embedded,cn_embedded),dim=-1),
                            src_mask,
                            src_lengths)
    
    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None,
               cn=None):
        
        embedded = self.trg_embed(trg) # concatenate country embeddings with token embeddings
        
        batch_size = embedded.size(0)
        sequence_size = embedded.size(1)
        cn_embedded = self.cn_embed(cn)
        cn_embed_size = cn_embedded.size(1)
        assert batch_size == cn_embedded.size(0)
        cn_embedded = cn_embedded.unsqueeze(1).expand(batch_size,sequence_size,cn_embed_size).contiguous()  
        
        return self.decoder(torch.cat((embedded,cn_embedded),dim=-1),
                            encoder_hidden,
                            encoder_final,
                            src_mask,
                            trg_mask,
                            hidden=decoder_hidden)
        
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Classifier(nn.Module):
    """Define standard linear classifer"""
    def __init__(self,
                 hidden_size,
                 num_classes=3,
                 dropout=0.2):    
        super(Classifier, self).__init__()    
        # standard classifier layer
        self.classifier = nn.Sequential(nn.Linear(2 * hidden_size, 300),
                                        nn.Dropout(p=dropout),
                                        nn.LeakyReLU(),
                                        nn.Linear(300, 128),
                                        nn.Dropout(p=dropout),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        # return F.log_softmax(self.classifier(x).mean(dim=1),dim=-1)
        return self.classifier(x).mean(dim=1)
    
class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final
    
class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas    
    
class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, heavy_decoder=False,
                 add_input_skip=False, skip_attention=None):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.add_input_skip = add_input_skip
        
        if add_input_skip:
            self.skip_attention = skip_attention
        
        if heavy_decoder:
            # the idea is not to create a bottleneck in the representation
            # we have biGRU in the encoder vs GRU in the decoder 
            self.rnn = nn.GRU(emb_size + 2*hidden_size, 2*hidden_size, num_layers,
                              batch_first=True, dropout=dropout)

            # to initialize from the final encoder state
            self.bridge = nn.Linear(2*hidden_size, 2*hidden_size, bias=True) if bridge else None
            
            self.dropout_layer = nn.Dropout(p=dropout)
            self.pre_output_layer = nn.Linear(2*hidden_size + 2*hidden_size + emb_size*(1+add_input_skip),
                                              hidden_size, bias=False)            
        else:
            self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
                 
            # to initialize from the final encoder state
            self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

            self.dropout_layer = nn.Dropout(p=dropout)
            self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size*(1+add_input_skip),
                                              hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden,
                     skip=None, skip_key=None):
        """Perform a single decoder step (1 word)"""
        
        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)
        
        if self.add_input_skip:
            context_skip, attn_probs_skip = self.attention(
                query=query, proj_key=skip_key,
                value=skip, mask=src_mask)            
        
        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        if self.add_input_skip:
            # also add input sequence embedding
            pre_output = torch.cat([prev_embed, output, context, context_skip], dim=2)
        else:
            pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None, skip=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        if self.add_input_skip:
            # also for the skip connection
            skip_key = self.skip_attention.key_layer(skip)
        else:
            skip_key = None
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden,
                skip=skip, skip_key=skip_key)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))