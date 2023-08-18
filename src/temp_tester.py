from api.prediction import predict

y_pred = predict(
    images='/home/matheust/projects/age-prediction/data/test/21336_35_1_0.jpg',
    model_type='regular',
    coverage=0.9
)

print(y_pred)
