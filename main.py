from csvhandler import CsvHandler
from os import listdir

def list_csv():
	ret = []
	for filename in listdir():
		if filename.lower().endswith(".csv"):
			ret.append(filename)
	return ret

def select_csv():
	csvs = list_csv()
	for i in range(len(csvs)):
		print(i, ")", csvs[i])
	s = input("Load file, number:")
	try:
		return csvs[int(s)]
	except Exception:
		return ""

def intlist(list):
	ret = []
	for item in list:
		ret.append(int(item))
	return ret

def get_target_and_features_from_user(c):
	print(c.column_numbers_and_names())
	target = input("Target (=y) column number:")
	f = input("Feature (=X) column number(s), separated by ',':")
	features = f.split(',')
	return target,features

def list_columns(c):
	print(c.list_columns())

def histogram(c):
	print(c.column_numbers_and_names())
	col_number = input("Column number:")
	try:
		c.histogram(col_number)
	except Exception as e:
		print(e)

def load_file(c):
	f = select_csv()
	d = input("Delimiter (default=','):")
	c.load_file(f,d)
	print(c.head())

def plot_scatter(c):
	target,features = get_target_and_features_from_user(c)
	try:
		c.plot_scatter(int(target), intlist(features))
	except Exception as e:
		print(e)

def linear_regression(c):
	target,features = get_target_and_features_from_user(c)
	try:
		c.linear_regression(int(target), intlist(features))
	except Exception as e:
		print(e)

def svc(c, kernel_type):
	target,features = get_target_and_features_from_user(c)
	try:
		c.svc(int(target), intlist(features), kernel_type)
	except Exception as e:
		print(e)

def knn(c):
	target,features = get_target_and_features_from_user(c)
	try:
		c.knn(int(target), intlist(features))
	except Exception as e:
		print(e)

def logreg(c):
	target,features = get_target_and_features_from_user(c)
	try:
		c.logreg(int(target), intlist(features))
	except Exception as e:
		print(e)

def print_help():
	print("---")
	print("help: print this help")
	print("columns: list columns and their data types")
	print("head: print csv head information")
	print("heatmap: show heatmap for correlations")
	print("histogram: draw histogram")
	print("scatter: plot scatter diagram")
	print("linreg: (regression) use linear regression model")
	print("svcrbf: (classification) use svc model with rbf kernel")
	print("svclinear: (classification) use svc model with linear kernel")
	print("logreg: (classification) use logistic regression model")
	print("knn: (classification) use k nearest neighbors model")
	print("load: load new csv file")
	print("quit: quit program")
	print("---")

def main():
	c = CsvHandler()

	while True:
		if c.filename() == "":
			load_file(c)
		else:
			rows, columns = c.shape()
			print("Using file '" + c.filename() + "', "+ str(rows) + " rows," + " " + str(columns) + " columns")
			print("Available commands: help, columns, head, heatmap, histogram, scatter, linreg, svcrbf, svclinear, logreg, knn, load, quit")
			s = input("command:")
			if s == "help":
				print_help()
			elif s == "columns":
				list_columns(c)
			elif s == "head":
				print(c.head())
			elif s == "heatmap":
				c.show_heatmap()
			elif s == "histogram":
				histogram(c)
			elif s == "scatter":
				plot_scatter(c)
			elif s == "linreg":
				linear_regression(c)
			elif s == "svcrbf":
				svc(c, "rbf")
			elif s == "svclinear":
				svc(c, "linear")
			elif s == "logreg":
				logreg(c)
			elif s == "knn":
				knn(c)
			elif s == "load":
				load_file(c)
			elif s == "quit":
				break

if __name__ == "__main__":
	main()
