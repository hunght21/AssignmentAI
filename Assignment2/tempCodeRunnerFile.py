import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ tệp
data = np.loadtxt('C:/Users/ACER/OneDrive - Hanoi University of Science and Technology/Documents/Desktop/AI-2023/AssAI/Assignment2/svmTuningData.dat', delimiter=',')

# Tách dữ liệu thành đặc trưng và nhãn
X = data[:, :-1]  # Lấy tất cả các cột trừ cột nhãn
y = data[:, -1]   # Lấy cột nhãn

# Tách dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tìm kiếm qua không gian tham số để tìm giá trị tối ưu cho C và sigma
param_grid = {'C': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100],
              'gamma': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]}

grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# In ra giá trị tối ưu
optimal_C = grid_search.best_params_['C']
optimal_sigma = grid_search.best_params_['gamma']

# Đào tạo mô hình SVM với các tham số tối ưu
best_svm_model = grid_search.best_estimator_
best_svm_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm thử
y_pred = best_svm_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)


print("SVM_OPTIMAL_C:", optimal_C, "\n")

print("SVM_OPTIMAL_SIGMA:", optimal_sigma, "\n")

print("SVM_OPTIMAL_ACCURACY:", accuracy, "\n")
