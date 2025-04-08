import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

class myBertModel(nn.Module):
    def __init__(self, bert_path, num_class, device):
        super(myBertModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)
        # config = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel(config)

        self.device = device
        self.cls_head = nn.Linear(768, num_class)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)



    def forward(self, text):
        input = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        input_ids = input["input_ids"].to(self.device)
        token_type_ids = input["token_type_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)
        sequence_out, pooler_out = self.bert(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        return_dict=False)

        pred = self.cls_head(pooler_out)
        return pred


if __name__ == "__main__":
    model = myBertModel("../bert-base-chinese", 2)
    pred = model("今天天气真好")


















