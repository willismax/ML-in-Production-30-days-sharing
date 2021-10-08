# [Day 19 : 深度學習(神經網絡)自動建模術 - AutoMLs](https://ithelp.ithome.com.tw/articles/10266499)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10266499)

-   隨著 ML/DL 模型研究屢有突破，現今模型訓練成果已經相當具有水準，但如果需要藉由手動選擇最佳的模型確實較花時間，因此已經出現取多自動化機械學習 AutoMLs 方案，雲端平台 Azure 、AWS、GCP 也提供許多 AutoMLs 服務，並主打輕鬆解決 ML 複雜的自動建構機械學習與預測服務。
-   不過對於 AutoMLs 應有個基本認知，AutoMLs 會遴選多種較通用的模型取得最佳成績，但如果您有特別的模型需求，AutoMLs不見得可以提供最適切的效果，在自動建模的搜尋過程中，多採較泛用的模型進行選擇，您可以期待得到與人類水平相近似的成績，但要追求頂尖成果您可以接手自行嘗試。
-   AutoMLs 中討論自動化神經網絡搜尋的子集稱之為 Neural Architecture Search (NAS) ，是實現以自動化搜尋神經網路最佳解決方案的技術。
-   也因為 AutoMLs 的出現， 追求自動的 MLOps 解決方案以因應持續訓練 Continuous Training (CT) 成為可能。以下分別介紹基於雲端平台的 AutoMLs 工具，以及開源的 Auto Keras。

基於雲端平台的 AutoMLs 工具
------------------

-   採取公有雲的方案最大好處是「用多少算多少」，省下大量購置/閒置/折舊成本開銷，當然量一多燒錢也很可觀。

### Amazon SageMaker Autopilot

-   [Amazon SageMaker Autopilot](https://aws.amazon.com/sagemaker/autopilot) 能依據資料來自動建置、訓練和調整最佳 ML 模型，僅需提供表格式資料集並選取目標直欄即可進行預測，它可以是數值 (例如：房價，稱為迴歸)，或類別 (例如：垃圾郵件/非垃圾郵件，稱為分類)。
    -   ![](https://d1.awsstatic.com/SageMaker/SageMaker%20reInvent%202020/Autopilot/product-page-diagram_SageMaker_Auto-Pilot_dk-bg%402x.e2d27caf8ec3224f1498d904aee630f61c847359.png)

### Microsoft Azure Automated Machine Learning

-   [Microsoft Azure Automated Machine Learning](https://azure.microsoft.com/en-in/services/machine-learning/automatedml/) 強調以速度和規模自動構建機器學習模型，結合微軟雲端平台優勢以文件處理的能耐，創造出不錯的使用體驗。
    -   ![](https://azure.microsoft.com/cdn/cvt-9c98bcf08ba179a48076ed1ac915e0b03792ce1515d95049f1e7f94fa10547e9/images/page/services/machine-learning/automatedml/learning-models.gif?637668606505980159)

### Google Cloud AutoML

-   [Google Cloud AutoML](https://cloud.google.com/automl) 說明即使您對於機器學習領域並沒有充足的專業知識，也能輕鬆訓練出品質優異的自訂機器學習模型，在許多解決方案中，[Vertex AI](https://cloud.google.com/vertex-ai/docs) 將 AutoML 和 AI Platform 整合到一個統一的 API、客戶端庫和用戶界面中。

開源的解決方案 AutoKeras
-----------------

![](https://camo.githubusercontent.com/1b4dfa29a12e42177feb68289fb3954069dac657021996ef09a8c182737bdf03/68747470733a2f2f6175746f6b657261732e636f6d2f696d672f726f775f7265642e737667)

-   [AutoKeras](https://autokeras.com/)：基於 Keras 的 AutoML 系統。它由德克薩斯農工大學的 [DATA 實驗室](http://faculty.cs.tamu.edu/xiahu/index.html)開發。AutoKeras 的目標是讓每個人都可以使用機器學習。
    
-   AutoKeras 的社群相當活躍，您可以找到許多解決方案。
    
-   AutoKeras 目前支持的 Auto MLs 包含:
    
    -   圖像分類`ImageClassifier`
    -   圖像回歸`ImageRegressor`
    -   文本分類`TextClassifier`
    -   文本回歸`TextRegressor`
    -   結構化資料分類`StructuredDataClassifier`
    -   結構化資料回歸`StructuredDataRegressor`
    -   時間序列預測`TimeseriesForecaster`
    -   多模式任務
-   上述[官方範例文件](https://autokeras.com/tutorial/overview/)及對應可執行的 Colab ，另外文件也指出，如果需要更加進階、客製自動搜尋模型，可以用`AutoModel`以及其參數實現，歡迎您嘗試。
    

### 使用介紹

以下以可以發現與一般訓練流程差不多，以下以官方範例[Image Classification](https://autokeras.com/tutorial/image_classification/) 說明， [官方 Colab 範例 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/keras-team/autokeras/blob/master/docs/ipynb/image_classification.ipynb) 供參考，但提醒您會跑超過半小時，您要測試時建議減少 epochs 等參數。

-   安裝 `autokeras` 模組。
    
    ```python
    !pip install autokeras
    
    ```
    
-   訓練模型(摘述)，與 `tf.keras` 訓練方式類似。
    
    ```python
    import autokeras as ak
    
    # 初始化實例
    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
    
    # 訓練模型，會比較花時間，跑著跑著最佳模型與參數就出來了。
    clf.fit(x_train, y_train, epochs=10)
    
    
    ```
    
-   最終結果  
    ![](https://i.imgur.com/jmazSfT.png)
    
-   您可以注意到，最佳模型結果已經儲存在`./image_classifier/best_model/` 之中，包含`save_model.pb` 與 `keras_metadata.pb` 中繼資料。  
    ![](https://i.imgur.com/Vlhvhcl.png)
    

小結
--

-   AutoMLs 自動搜尋最佳模型解決方案，可以省下人工調參及選擇模型的時間，時間就是金錢。
-   主流雲端服務廠商皆有提供 AutoMLs 的整合服務，開源的 Auta Keras 也很活躍可以運用。

參考
--

-   [Amazon SageMaker Autopilot](https://aws.amazon.com/sagemaker/autopilot)
-   [Microsoft Azure Automated Machine Learning](https://azure.microsoft.com/en-in/services/machine-learning/automatedml/)
-   [Google Cloud AutoML](https://cloud.google.com/automl)
-   [AutoKeras](https://autokeras.com/)
