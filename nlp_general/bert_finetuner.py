
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import math, os
from datasets import load_dataset
from configs.params import MATERIALS_DIR

def finetune_mlm(cond):

    print("Finetuning MLM !!!! ")
    os.environ["WANDB_DISABLED"] = "true"
    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb#scrollTo=iAYlS40Z3l-v
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples, block_size = 128):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    text_file = f"{MATERIALS_DIR}/transcripts/milkyway{cond}_transcript.txt"
    model_checkpoint = "bert-base-uncased"

    datasets = load_dataset("text", data_files={"train": text_file, "validation": text_file})
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # model = base_model#
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    # model_name = "bert-base-uncased"
    training_args = TrainingArguments(
        "test-clm",
        # place_model_on_device = False,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=False,
        # push_to_hub_model_id=f"{model_name}-finetuned-wikitext2",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    trainer = Trainer(
        model=model,
        # place_model_on_device = False,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        # plac
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    return model, tokenizer


