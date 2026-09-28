def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def get_probabilities(model, X_val):
    return model.predict_proba(X_val)[:, 1]  # 取「有糖尿病」的機率