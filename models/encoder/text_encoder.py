import transformers


from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable, config):
        super().__init__()

        self.model_name = config['text_encoder']['model_name']
        self.pretrained = config['global']['pretrained']
        self.trainable = config['global']['trainable']
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]