import torch
import torch.nn as nn
import numpy as np

START_TAG = '<START>'
STOP_TAG = '<STOP>'

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25,  #25
                 char_embedding_dim=25, char_lstm_dim=25, dropout=0.5, 
                 use_gpu=False, use_crf=True, char_mode='CNN', device=None):
        
        super(BiLSTM_CRF, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode
        self.char_lstm_dim = char_lstm_dim

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.init_embedding(self.word_embeds.weight)

        if char_to_ix is not None:
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            self.init_embedding(self.char_embeds.weight)

            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, 
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))
                
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                self.init_lstm(self.char_lstm)
            
        
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)

        self.init_lstm(self.lstm)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        self.init_linear(self.hidden2tag)

        if self.use_crf:
            self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
            
        self.to(self.device) 

    def init_lstm(self, input_lstm):
        for name, param in input_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.zero_()
                n = input_lstm.hidden_size
                param.data[n:2*n] = 1.0

    def init_linear(self, input_linear):
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias) # FIX: uniform_
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def init_embedding(self, input_embedding):
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform_(input_embedding, -bias, bias) 

    def _get_lstm_features(self, sentence, chars2, chars2_length, d):
        if self.char_mode == 'LSTM':
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            
            chars_embeds_temp = torch.zeros((outputs.size(0), outputs.size(2))).to(self.device)
            
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], 
                                                outputs[i, 0, self.char_lstm_dim:]))
            
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]
        
        elif self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        word_embeds = self.word_embeds(sentence)
        embeds = torch.cat((word_embeds, chars_embeds), 1)
        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)

    def _score_sentence(self, feats, tags):
        r = torch.arange(feats.size()[0]).to(self.device)
        pad_start_tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]]).to(self.device), tags])
        pad_stop_tags = torch.cat([tags, torch.tensor([self.tag_to_ix[STOP_TAG]]).to(self.device)])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
        return score

    def _forward_alg(self, feats):
        alpha = torch.full((1, self.tagset_size), -10000., device=feats.device)
        alpha[0][self.tag_to_ix[START_TAG]] = 0.

        for feat in feats:
            next_tag_var = alpha + self.transitions + feat.view(-1, 1)
            alpha = torch.logsumexp(next_tag_var, dim=1).view(1, -1)

        terminal_var = alpha + self.transitions[self.tag_to_ix[STOP_TAG]]
        return torch.logsumexp(terminal_var, dim=1)

    def viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().detach().cpu().numpy()
            
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var[self.tag_to_ix[START_TAG]] = -10000.
        
        best_tag_id = torch.argmax(terminal_var).item()
        path_score = terminal_var[best_tag_id]
        
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(int(best_tag_id))
            
        best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence, chars, chars2_length, d):
        feats = self._get_lstm_features(sentence, chars, chars2_length, d)
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = tag_seq.cpu().tolist()

        return score, tag_seq

    def neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        feats = self._get_lstm_features(sentence, chars2, chars2_length, d)
        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            return nn.functional.cross_entropy(feats, tags)