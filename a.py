from dotenv import load_dotenv
import os, requests

load_dotenv()
key = os.getenv("DATA_GO_KR_SERVICE_KEY")

url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

params = {
    "serviceKey": key,
    "pageNo": 1,
    "numOfRows": 1000,
    "dataType": "JSON",
    "base_date": "20260523",
    "base_time": "2100",
    "nx": 58,
    "ny": 74,
}

r = requests.get(url, params=params)
print(r.status_code)
print(r.text[:1000])
