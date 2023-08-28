from api.prediction import predict

y_pred = predict(
    images='data/test/8802_53_0_0.jpg',
    model_type='regular'
)

print(y_pred)
