gpx-navi 

gpxconverter.py
ルート作成
stravaのルートを.gpxでエクスポートする
そのgpxには、wpt（曲がり角）の情報が入っていないので、それを追加するあぷり
streamlitで動かす

streamlit run .\gpxconverter.py

index.html
ナビ本体
gpxconverterで作成したターン情報付きのgpxを読み込んでナビする
iphoneのSafariでの動作を前提
（Debug中はsafariのキャッシュを消しながらやらないと、前のバージョンが残り、わけわからなくなるので注意）
github上で公開。あどれすは
https://gtatsu33.github.io/gpx-navi/

git add .
git commit -m "comment"
git push origin main

