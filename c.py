import pandas as pd

df = pd.read_csv(
    "data/solar_api.csv",
    low_memory=False
)

# 나주 주소 필터
naju_mask = (
    df["lctnRoadNmAddr"].astype(str).str.contains("광주", na=False)
    |
    df["lctnLotnoAddr"].astype(str).str.contains("광주", na=False)
)

# 위도/경도 존재 여부
coord_mask = (
    df["latitude"].notna()
    &
    df["longitude"].notna()
)

# 둘 다 만족
result = df[naju_mask & coord_mask]

print("나주 + 위경도 존재 데이터 수:", len(result))

# 확인용 출력
print(
    result[
        [
            "solarGenFcltNm",
            "lctnRoadNmAddr",
            "lctnLotnoAddr",
            "latitude",
            "longitude"
        ]
    ].head(20)
)