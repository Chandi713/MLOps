from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, x_train, y_train, x_test, y_test):
   
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Metrics
    return{
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }