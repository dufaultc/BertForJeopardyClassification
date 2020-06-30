from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, glue_convert_examples_to_features, InputExample, InputFeatures, TrainingArguments, Trainer, DefaultDataCollator
from torch.utils.data.dataset import Dataset
import json

with open('JEOPARDY_QUESTIONS1.json') as f:
    data = json.load(f)
loadconfig = BertConfig.from_pretrained('bert-base-uncased', num_labels = 3, finetuning_task ='text-classification')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = loadconfig)
class CustomDataset(Dataset):
    def __init__(self, tokenizer, question_list, block_size: int):     
        real_examples = []
        i = 0
        self.labels = []
        for question in question_list:
            if question['round'] == 'Jeopardy!':
                self.labels.append(0)
            elif question['round'] == 'Double Jeopardy!':
                self.labels.append(1)                
            else:
                self.labels.append(2)
            real_examples.append(InputExample(guid = i, text_a = question['question'], label = self.labels[i]))
            i=i+1
        self.features = glue_convert_examples_to_features(examples = real_examples, tokenizer = tokenizer, max_length = 200, label_list = [0,1,2], output_mode = 'classification')
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index) -> InputFeatures:
        return self.features[index]
    
train_dataset = CustomDataset(
    tokenizer=tokenizer,
    question_list=data,
    block_size = 2004
)

training_args = TrainingArguments(
    output_dir="./Models/JeopardyFineTuning",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_gpu_train_batch_size=8,
    save_steps=50,
    save_total_limit=2,
    do_train = True,
)

trainerfine = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator = DefaultDataCollator()
)

trainerfine.train()