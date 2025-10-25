# Finance Sentiment Analyzer (BERT + Streamlit)

This repository contains code to:
- Fine-tune BERT on the Financial PhraseBank dataset using TensorFlow.
- Run inference on live news fetched with `gnews`.
- Display results in a Streamlit app.

## File overview
- `train_sentiment_bert.py`: Train and save a BERT model (TensorFlow).
- `inference.py`: Load the saved model/tokenizer and provide `predict_sentiment`.
- `streamlit_app.py`: Streamlit app that fetches news and shows predictions.
- `data_download_and_prepare.py`: Download Financial PhraseBank to CSV for inspection.
- `requirements.txt`: Python dependencies.

## Quick start
1. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS / Linux
   venv\Scripts\activate      # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download dataset CSV:
   ```bash
   python data_download_and_prepare.py
   ```
4. Train the model (recommended on GPU or Colab):
   ```bash
   python train_sentiment_bert.py
   ```
   This creates `saved_model/tf_model` and `saved_model/tokenizer`.
5. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Notes
- Training BERT requires substantial compute. If you don't have a GPU, reduce `EPOCHS` or use a smaller model like DistilBERT.
- `gnews` is used for prototyping. For production, consider NewsAPI or other paid sources.
- Later you can swap in FinBERT or other finance-tuned models for better performance.
