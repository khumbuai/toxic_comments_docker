from src.models.LSTM_MultiAttention.predict import ToxicMultiAttentionModel

predictor = ToxicMultiAttentionModel(dir='src/models/LSTM_MultiAttention/')


def predict_on_text(raw_text):
    try:
        predictions = predictor.predict(raw_text)
        return predictions
    except Exception as e:
        print(e)
