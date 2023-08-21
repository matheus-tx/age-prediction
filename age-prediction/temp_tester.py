from api.prediction import predict

y_pred = predict(
    images='/home/matheust/projects/age-prediction/data/test/18965_20_1_0.jpg',
    model_type='regular'
)

print(y_pred)
