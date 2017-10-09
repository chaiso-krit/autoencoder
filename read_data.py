class StockData:
    def __init__(self, open, high, low, close, volume):
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = int(volume)
    
    def __str__(self):
        return 'Open:' + str(self.open) + ',Close:' + str(self.close) + ',Volume:' + str(self.volume)

class Data:
    start_date = 0
    data_range = 60
    stock_directory = 'dataset/stock_only/'

    def __init__(self, name):
        self.data = []
        fptr = open(self.stock_directory+name, 'r')
        for line in fptr:
            data = line.split(',')
            stockData = StockData(data[1],data[2],data[3],data[4],data[5])
            self.data.append(stockData)
        fptr.close()
        self.current_date = 0

    def size(self):
        return len(self.data)

    def get_first(self):
        self.current_date = self.start_date + self.data_range
        return self.data[self.start_date:self.start_date + self.data_range]

    def get_next(self):
        self.current_date += 1
        return self.data[self.current_date-1]

    def get_day_left(self):
        return self.size() - self.current_date 

if __name__ == "__main__":
    data = Data('ABICO')   
    for i in range(100):
        print data.get_next()