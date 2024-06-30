# BLASTAM Japan

BLASTAM Japanは、BLASTAMモデルを使用して稲熱病（いもち病）のリスク評価を視覚的に表示するためのウェブベースのツールです。このツールは、リスク評価スコアを色分けされたマーカーで地図上に表示し、研究者や農家にとってわかりやすいビジュアル化を提供します。
BLASTAM Japan is a web-based tool designed to visually present the risk assessment of rice blast disease (稲熱病、いもち病) using the BLASTAM model. The tool displays the risk assessment scores on a map with color-coded markers, providing an easy-to-understand visualization for researchers and farmers.

## 特徴 / Features

**インタラクティブな地図**：日本各地の稲熱病リスク評価の視覚的表示。
**Interactive Map**: Visual representation of rice blast disease risk assessment across various locations in Japan.

**日付スライダー**：過去20日間のリスク評価を選択して表示することができます。
**Date Slider**: Allows users to select and view the risk assessment for the past 20 days.

**詳細情報**：各マーカーをクリックすると、気象観測所の詳細情報と過去20日間のリスク評価スコアが表示されます。
**Detailed Information**: Clicking on each marker provides detailed information about the weather station and the risk assessment scores for the past 20 days.

**モデル選択**：将来の複数モデルの実装のためのプレースホルダー。
**Model Selection**: Placeholder for future implementation of multiple models.

## 使い方 / Usage

1. **モデル選択**：ドロップダウンメニューからモデル（現在はBLASTAMのみ）を選択します。
   **Select Model**: Choose the model (currently only BLASTAM) from the dropdown menu.

2. **日付スライダーの調整**：下部の日付スライダーを使用して、リスク評価を表示する日付を選択します。
   **Adjust Date Slider**: Use the date slider at the bottom to select the date for which you want to view the risk assessment.

3. **マーカーをクリック**：地図上のマーカーをクリックして、気象観測所の詳細情報と過去20日間のリスク評価を表示します。
   **Click Markers**: Click on the markers on the map to view detailed information about the weather station and the risk assessment for the past 20 days.

## カラースキーム / Color Codes

**赤色**：「好適条件」を示します。
**Red**: High risk ("好適条件")

**黄色**：「準好適条件」を示します。
**Yellow**: Medium risk ("準好適条件")

**緑色**：「非好適条件」を示します。
**Green**: Low risk ("非好適条件")

## データソース / Data Sources

**気象データ**：[AMeDAS](https://www.data.jma.go.jp/gmd/risk/obsdl/) から取得。
**Weather Data**: Retrieved from [AMeDAS](https://www.data.jma.go.jp/gmd/risk/obsdl/)

**計算公式**：越水幸男. 1988. 「アメダス資料による葉いもち発生予察法」。 [Agriknowledge](https://agriknowledge.affrc.go.jp/RN/2030411788)
**Reference for Calculation**: Y. Koshimizu. 1988. "Prediction method of rice blast occurrence using AMeDAS data". [Agriknowledge](https://agriknowledge.affrc.go.jp/RN/2030411788)

## 追加情報 / Additional Information

このウェブページのリポジトリは [BLASTAM Japan](https://github.com/Raingel/BLASTAM_Japan) にあります。
The repository for this web page is available at [BLASTAM Japan](https://github.com/Raingel/BLASTAM_Japan).

計算公式は次のリファレンスから引用されています: 越水幸男. 1988. 「アメダス資料による葉いもち発生予察法」。 [Agriknowledge](https://agriknowledge.affrc.go.jp/RN/2030411788)
The calculation formula is referenced from: Y. Koshimizu. 1988. "Prediction method of rice blast occurrence using AMeDAS data". [Agriknowledge](https://agriknowledge.affrc.go.jp/RN/2030411788)

実装は [BLASTAM](https://github.com/Raingel/BLASTAM) によるものです。
The implementation is by [BLASTAM](https://github.com/Raingel/BLASTAM).

気象データは [AMeDAS](https://www.data.jma.go.jp/gmd/risk/obsdl/) から取得しています。
Weather data is sourced from [AMeDAS](https://www.data.jma.go.jp/gmd/risk/obsdl/).

赤色は「好適条件」を示します。
Red indicates "high risk conditions".

黄色は「準好適条件」を示します。
Yellow indicates "medium risk conditions".

緑色は「非好適条件」を示します。
Green indicates "low risk conditions".

## コントリビューション / Contributing

機能や特徴を強化するための貢献を歓迎します。リポジトリをフォークしてプルリクエストを送信してください。
We welcome contributions to enhance the functionality and features of this project. Please feel free to fork the repository and submit pull requests.

## 参考文献 / References

- Ou, J. H., Kuo, C. H., Wu, Y. F., Lin, G. C., Lee, M. H., Chen, R. K., ... & Chen, C. Y. (2023). Application-oriented deep learning model for early warning of rice blast in Taiwan. Ecological Informatics, 73, 101950.
- Koshimizu, Y. 1988. "Prediction method of rice blast occurrence using AMeDAS data". [Agriknowledge](https://agriknowledge.affrc.go.jp/RN/2030411788)
- Weather data provided by [AMeDAS](https://www.data.jma.go.jp/gmd/risk/obsdl/)
