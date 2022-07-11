from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
hsv_pred=np.load('/content/gdrive/MyDrive/Prediction_Arrays_HandGesture/mymodel_NUS_HSV_1.npy')
yuv_pred=('/content/gdrive/MyDrive/Prediction_Arrays_HandGesture/mymodel_NUS_YUV_1.npy')
y_test_1d = encode_y(y)
from sklearn.metrics import accuracy_score
# models = [model_RGB, model_HSV]
preds = [hsv_pred, yuv_predict]
preds = np.array(preds)
# sum rule
summed = np.sum(preds, axis=0)
ensemble_prediction = np.argmax(summed, axis=1)
ensembled_accuracy = accuracy_score(ensemble_prediction, y_test_1d)
ensembled_accuracy
