from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("AlekseyKorshuk/persona-chat", split="train")
eval_dataset = load_dataset("AlekseyKorshuk/persona-chat", split="validation")  # Correct split

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Set pad token to eos token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Prepare dialogues by combining the personality and utterances
    dialogues = [
        ' '.join(personality + utterance['history'] + [response])
        for personality, utterances in zip(examples['personality'], examples['utterances'])
        for utterance in utterances  # Loop through all utterances
        for response in utterance['candidates'][:1]  # Use the first candidate response
    ]
    
    # Tokenize dialogues with padding and truncation
    tokenized = tokenizer(dialogues, padding='max_length', truncation=True, max_length=512)

    # Add labels (shifting the input_ids by one token)
    input_ids = tokenized['input_ids']
    labels = input_ids.copy()  # Make a copy of input_ids to serve as labels
    for i in range(len(input_ids)):
        labels[i] = [-100] + input_ids[i][:-1]  # Shift labels by one token (and use -100 for padding tokens)
    
    # Return tokenized data with labels
    return {
        'input_ids': input_ids,
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    }

# Apply preprocessing
tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=['personality', 'utterances'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",     
    learning_rate=2e-5,              
    per_device_train_batch_size=4,   
    num_train_epochs=3,              
    save_total_limit=2,              
    logging_dir='./logs',            
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=tokenized_data,        
    eval_dataset=eval_dataset,           # Pass the eval dataset
    tokenizer=tokenizer                  
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_persona_chatbot")
tokenizer.save_pretrained("fine_tuned_persona_chatbot")
