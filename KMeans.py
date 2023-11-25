import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import joblib


data = pd.read_parquet('data/train_dataset_hackaton2023_train.gzip')
LA = LabelEncoder()


# Загрузите сохраненную модель из файла
kmean = joblib.load('trained_models/kmeans_model.pkl')


def preproces(df):
    data = df.copy()
    data = data.drop(['group_name', 'dish_name'], axis=1)
    cost = data.groupby(['customer_id', 'startdatetime'], as_index=False).agg({'revenue':'sum'})
    data = data.drop('revenue', axis=1)
    data = data.merge(cost, on=['customer_id', 'startdatetime'], how='left')
    cost = data.groupby(['customer_id'], as_index=False).agg({'revenue':'sum'})
    data = data.rename(columns={'revenue':'check'})
    data = data.merge(cost, on=['customer_id'], how='left')
    data = data.rename(columns={'revenue':'sum'})
    data['dw'] = data.startdatetime.dt.day_of_week
    data['month'] = data.startdatetime.dt.month
    data['dy'] = data.startdatetime.dt.day_of_year
    data['hour'] = data.startdatetime.dt.hour
    data = data.sort_values(by='startdatetime', ascending=False)
    data = data.groupby('customer_id').head(1)
    data = data.drop('startdatetime', axis=1)
    data.format_name = LA.fit_transform(data.format_name)
    return data


data= preproces(data)
target = data.buy_post
data = data.drop(['date_diff_post', 'buy_post'], axis=1)


print(classification_report(target, kmean.labels_))
f1_score(target, kmean.labels_, average='binary')


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(target, kmean.labels_)
roc_auc = auc(fpr, tpr)

# Построение графика ROC-AUC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


test = pd.read_parquet('data/test_dataset_hackaton2023_test.gzip')


test = preproces(test)


result = kmean.predict(test)


sub = pd.read_csv('submission.csv', sep=';')


sub.buy_post = result


sub.to_csv('cluste.csv', sep=';', index=False)

