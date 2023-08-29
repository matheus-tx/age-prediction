from api.prediction import predict

y_pred = predict(
    images='data/test/19168_18_0_3.jpg',
    model_type='poisson',
    coverage=0.5
)

print(y_pred)
