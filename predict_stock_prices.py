import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plot


def get_data(filename):
    dates = []
    prices = []

    with open(filename, 'r') as csvFile:
        csv_file_reader = csv.reader(csvFile)
        next(csv_file_reader)
        for row in csv_file_reader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))

    return np.array(dates), np.array(prices)


def predict_prices(dates, prices, x):
    dates.dot(dates.transpose())
    dates.shape = (len(dates), 1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plot.scatter(dates, prices, edgecolors='black', data='Data')
    plot.plot(dates, svr_rbf.predict(dates))
    plot.plot(dates, svr_rbf.predict(dates))
    plot.plot(dates, svr_rbf.predict(dates))
    plot.xlabel('Date')
    plot.ylabel('Price')
    plot.title('Support Vector Regression')
    plot.legend()
    plot.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


dates, prices = get_data('./data_files/aapl.csv')

print("\n", "Dates: ", dates.size, "\n", "Prices: ", prices.size)

predicted_price = predict_prices(dates, prices, 29)
print(predicted_price)
