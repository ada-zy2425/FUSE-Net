import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from create_dataset import MOSI, MOSEI, CH_SIMS, PAD

class MSADataset(Dataset):
    def __init__(self, config):
        ds = config.data.lower()
        if ds == 'mosi': loader = MOSI(config)
        elif ds == 'mosei': loader = MOSEI(config)
        elif ds == 'ch-sims': loader = CH_SIMS(config)
        else: raise ValueError(f"Unknown dataset: {config.data}")

        self.data, self.word2id, self.pretrained_emb = loader.get_data(config.mode)
        self.len = len(self.data)
        
        if self.len > 0:
           
            _, v0, a0, _ = self.data[0][0]
            config.visual_size = v0.shape[1]
            config.acoustic_size = a0.shape[1]
            config.embedding_size = 0 

    def __len__(self): return self.len
    def __getitem__(self, idx): return self.data[idx]

def get_loader(config, shuffle=True):
    dataset = MSADataset(config)
    print(f"Loaded {config.data} {config.mode} set with {len(dataset)} samples.")
    
    bert_tok = AutoTokenizer.from_pretrained(config.bert_dir) if config.use_bert else None

    def collate_fn(batch):
       
        batch = sorted(batch, key=lambda x: x[0][1].shape[0], reverse=True)
        
        labels = torch.cat([torch.from_numpy(x[1]) for x in batch], dim=0)
        
       
        visual = pad_sequence([torch.FloatTensor(x[0][1]) for x in batch], batch_first=True)
        acoustic = pad_sequence([torch.FloatTensor(x[0][2]) for x in batch], batch_first=True)
        
       
        lengths = torch.LongTensor([x[0][1].shape[0] for x in batch])
        
       
        bs = bt = bm = None
        if config.use_bert and bert_tok:
            txts = [" ".join(x[0][3]) for x in batch]
            encodings = bert_tok(txts, padding=True, truncation=True, return_tensors="pt")
            bs, bm = encodings['input_ids'], encodings['attention_mask']
          
            bt = encodings.get('token_type_ids', None)
        
       
        sentences = pad_sequence([torch.LongTensor([0]) for x in batch], padding_value=PAD)
            
        return sentences, visual, acoustic, labels, lengths, bs, bt, bm

    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)