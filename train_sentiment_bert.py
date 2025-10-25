# train_sentiment_bert.py
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizerFast, TFBertForSequenceClassification, create_optimizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Config
HF_DATASET = "takala/financial_phrasebank"   # HF wrapper
SUBSET = "sentences_50agree"                 # common choice: 50% agreement
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = "saved_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1) Load dataset
    print("Loading dataset from Hugging Face...")
    ds = load_dataset(HF_DATASET, SUBSET)
    df = ds["train"].to_pandas()
    print("Dataset shape:", df.shape)

    # 2) Ensure labels are integers 0..2
    if df['label'].dtype == object or df['label'].dtype == 'str':
        label_map = {"negative":0, "neutral":1, "positive":2}
        df['label'] = df['label'].map(label_map)

    # 3) Train/test split
    train_df, val_df = train_test_split(df, test_size=0.12, stratify=df['label'], random_state=42)

    # 4) Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    def encode_texts(texts):
        return tokenizer(
            texts.tolist(),
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

    train_enc = encode_texts(train_df['sentence'])
    val_enc = encode_texts(val_df['sentence'])

    # 5) Create tf.data.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_enc),
        train_df['label'].values
    )).shuffle(10000).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_enc),
        val_df['label'].values
    )).batch(BATCH_SIZE)

    # 6) Build model (TF)
    print("Loading pretrained BERT model...")
    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 7) Optimizer & compile
    steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)
    total_train_steps = steps_per_epoch * EPOCHS
    optimizer, schedule = create_optimizer(
        init_lr=3e-5,
        num_warmup_steps=int(0.1 * total_train_steps),
        num_train_steps=total_train_steps
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 8) Callbacks
    checkpoint_cb = ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, "best_model.h5"),
                                    monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
    early_cb = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

    # 9) Train
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_cb]
    )

    # 10) Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "tf_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    print("Model and tokenizer saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
