import urllib.request
import json


def main():
    url = "https://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20weather.forecast%20where%20woeid%20in%20(select%20woeid%20from%20geo.places(1)%20where%20text%3D%22nome%2C%20ak%22)&format=json&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys"
    data = urllib.request.urlopen(url).read()

    jsondata = json.loads(data.decode("utf-8"))
    print(jsondata["query"]["results"]["channel"])


if __name__ == '__main__':
    main()
