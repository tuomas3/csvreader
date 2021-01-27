import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

class CsvHandler:
	__filename = ""
	__df = pd.DataFrame()

	def head(self):
		return self.__df.head()

	def load_file(self, filename, delimiter):
		if delimiter == "":
			delimiter = ','
		try:
			self.__df = pd.read_csv(filename, delimiter=delimiter)
			self.__filename = filename
		except Exception as e:
			print(e)

	def filename(self):
		return self.__filename

	def get_data(self, target_column_number, feature_column_numbers):
		y = self.__df.iloc[:, target_column_number]
		X = self.__df.iloc[:, feature_column_numbers]
		X = self.convert_strings_to_numeric(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
		return X,X_train,X_test,y,y_train,y_test

	def shape(self):
		return self.__df.shape

	def show_heatmap(self):
		sns.heatmap(data=self.__df.corr().round(2), annot=True)
		plt.show()

	def list_columns(self):
		list = self.__df.columns.tolist()
		ret = []
		for i in range(len(list)):
			ret.append(str(i) + ") " + list[i] + " [" + str(self.__df[list[i]].dtype) + "]")
		return ret

	def plot_scatter(self, y_column_number, x_column_numbers):
		for i in range(len(x_column_numbers)):
			plt.subplot(1, len(x_column_numbers), i+1)
			plt.scatter(self.__df.iloc[:, x_column_numbers[i]], self.__df.iloc[:, y_column_number])
			plt.xlabel(self.__df.columns[x_column_numbers[i]])
			plt.ylabel(self.__df.columns[y_column_number])
		plt.show()

	def histogram(self, col_number):
		plt.hist(self.__df.iloc[:, int(col_number)])
		plt.show()

	def linear_regression(self, target_column_number, feature_column_numbers):
		X,X_train,X_test,y,y_train,y_test = self.get_data(target_column_number,feature_column_numbers)

		lm = LinearRegression()
		lm.fit(X_train, y_train)

		y_train_predict = lm.predict(X_train)
		rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
		r2 = r2_score(y_train, y_train_predict)
		print("RMSE=", rmse, "R2=", r2)

		y_test_predict = lm.predict(X_test)
		rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
		r2_test = r2_score(y_test, y_test_predict)
		print("RMSE (test)=", rmse_test, "R2 (test)=", r2_test)

		self.plot_linear_regression(X, X_train, X_test, y, y_train, y_test)

	def plot_linear_regression(self, X, X_train, X_test, y, y_train, y_test):
		for i in range(len(X.columns)):
			plt.subplot(1, len(X.columns), i + 1)
			plt.scatter(X.iloc[:, i], y)
			lm = LinearRegression()
			lm.fit(pd.DataFrame(X_train.iloc[:, i]), y_train)
			plt.plot(pd.DataFrame(X_test.iloc[:, i]), lm.predict(pd.DataFrame(X_test.iloc[:, i])), color="red")
			plt.xlabel(X.columns[i])
			plt.ylabel(y.name)
		plt.show()

	def svc(self, target_column_number, feature_column_numbers, kernel_type):
		X,X_train,X_test,y,y_train,y_test = self.get_data(target_column_number,feature_column_numbers)
		# if y.dtype == "float64":
		# 	y_train, y_test = self.discretize_target(y, y_train, y_test)
		svclassifier = SVC(kernel=kernel_type)
		svclassifier.fit(X_train, y_train)
		y_pred = svclassifier.predict(X_test)
		metrics.plot_confusion_matrix(svclassifier, X_test, y_test)
		plt.show()
		print("SVC", kernel_type, "kernel:")
		print(confusion_matrix(y_test, y_pred))
		print(classification_report(y_test, y_pred))

	def knn(self, target_column_number, feature_column_numbers):
		X,X_train,X_test,y,y_train,y_test = self.get_data(target_column_number,feature_column_numbers)
		classifier = KNeighborsClassifier(n_neighbors=3)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		metrics.plot_confusion_matrix(classifier, X_test, y_test)
		plt.show()
		cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
		print(cnf_matrix)
		print("KNN model, accuracy:", metrics.accuracy_score(y_test, y_pred))

	def logreg(self, target_column_number, feature_column_numbers):
		X, X_train, X_test, y, y_train, y_test = self.get_data(target_column_number, feature_column_numbers)
		# if y.dtype == "float64":
		# 	y_train, y_test = self.discretize_target(y, y_train, y_test)
		model = LogisticRegression()
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		metrics.plot_confusion_matrix(model, X_test, y_test)
		plt.show()
		cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
		print(cnf_matrix)
		print("Logistics regression model, accuracy:", metrics.accuracy_score(y_test, y_pred))

	def column_numbers_and_names(self):
		names = self.__df.columns.tolist()
		ret = ""
		for i in range(len(names)):
			ret += str(i) + ")" + names[i] + " "
		return ret

	def preprocess_with_label_encoder(self, X, X_train, X_test, y, y_train, y_test):
		enc = LabelEncoder()
		X_train_ret = pd.DataFrame()
		X_test_ret = pd.DataFrame()
		for column in X.columns:
			enc.fit(X[column])
			X_train_ret[column] = enc.transform(X_train[column])
			X_test_ret[column] = enc.transform(X_test[column])
		enc.fit(y)
		y_train_ret = enc.transform(y_train)
		y_test_ret = enc.transform(y_test)
		return X_train_ret, X_test_ret, y_train_ret, y_test_ret

	def convert_strings_to_numeric(self, X):
		for column in X.columns:
			if X[column].dtype == "object":
				# using .loc[] was suggested to avoid SettingWithCopyWarning, but the warning is still there (why?)
				# anyway, the result seems fine, if this operation is needed
				X.loc[:,column] = LabelEncoder().fit_transform(X.loc[:,column])
		return X

	def discretize_target(self, y, y_train, y_test):
		enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
		y_2d = pd.DataFrame(y) # encoder wants "2d" structure
		y_train_2d = pd.DataFrame(y_train)
		y_test_2d = pd.DataFrame(y_test)
		enc.fit(y_2d)
		y_train = enc.transform(y_train_2d)
		y_test = enc.transform(y_test_2d)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
		return y_train, y_test
