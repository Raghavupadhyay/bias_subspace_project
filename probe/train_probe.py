from sklearn.linear_model import LinearRegression

def train_probe(features, bias_scores):
    probe = LinearRegression()
    probe.fit(features, bias_scores)
    return probe.coef_