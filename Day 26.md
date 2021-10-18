# [Day 26 : 公平指標與實作 Fairness Indicators](https://ithelp.ithome.com.tw/articles/10271626)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10271626)


模型公平性的思考
--------

-   隨著 AI 對於各領域和社會的影響逐漸增加，建立公平且可包容所有人的系統至關重要，為達到負責任的 AI，重視公平性，實踐以人為本的設計初衷，確定多個指標來訓練與評估，檢查原始資料、理解資料與模型的局限性、部署後繼續監控和更新系統成了 MLOps 重要動力。
    
-   關於公平性沒有絕對的定義，不同國情、文化都有在意的事情，您可以參考[就業服務法第5條](https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=N0090001&flno=5)所示的就業平等認定，這是法律揭示應在就業情境中平等的規定，可用以檢視與追蹤您的服務模型與表達方式，以中性字眼表述，並持續調整。
    
-   平等不是指齊頭式的平等，而是客戶主觀意識認為接受到的服務是有受重視與在意本身需求。像是您不應該向素食主義者優先推薦葷食餐廳，宗教的飲食忌諱應降低推薦優先順序、以雙北生活圈習慣類推全台餐飲口味(北部粽、南部粽與...中部粽?)、推薦青少年不符年齡的遊戲清單、忽視生理性別與心理性別的服務推薦、將“護士”或“保姆”等詞翻譯成西班牙語時使用女性代詞等。以上現象在傳統數據分析、做研究追求整體準確率時不是考慮重點，卻是用於生產的機械學習服務必須要重視的關鍵任務。
    
    > 就業服務法第五條
    > 
    > -   為保障國民就業機會平等，雇主對求職人或所僱用員工，不得以種族、階級、語言、思想、宗教、黨派、籍貫、出生地、性別、性傾向、年齡、婚姻、容貌、五官、身心障礙、星座、血型或以往工會會員身分為由，予以歧視；其他法律有明文規定者，從其規定。
    
-   具體實踐機械學習系統的公平性來說，可以使用 [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide) 察覺模型數據在不同切片的表現，進行識別、改進模型。Google 的第二條 AI 原則指出，我們的技術應避免產生或強化不公平的偏見，提高模型的公平性。
    

可觀察公平性的指標
---------

### 陽性率、陰性率 positive, negative rate

-   基本的陽性或陰性的數據比率。
    -   應為獨立標籤，如果是平等的，子資料集也應該是平等的。可以用以檢視不同分組資料集的 PR 與 NR 比率應類似。
-   真陽率 true positive rate (TPR)，及假陰率 false negative rate (FPR)。
    -   真陽率 TPR 衡量在真實預測中被正確預測為陽性的陽性的比率。
    -   假陰率 FNR 衡量被錯誤預測為陰性的陽性數據比率。
    -   子資料集也應該有相同的 TPR 與 FNR 。

### 真假陽性率 TPR / FPR

-   真陰率 true negative rate (TNR) ，衡量實際為陰性、正確預測為陰性的比率。
-   假陽率 false positive rate (FPR) ，衡量實際為陰性、誤報為陽性的比率。
-   子資料集應該也有相似的 TNR 與 FPR。 FPR 誤報在分類的後續錯誤處置方式可能有不良影響，如果子資料集的 FPR 大於整體 FPR ，應該是值得關注的焦點。

### 準確率和 AUC

-   準確率 Accuracy 為判讀正確的比率， area under the curve (AUC) 為ROC曲線下與座標軸圍成的面積，最大值不大於1。
-   子資料集也應該與整體資料有相似的準確率與 AUC ，如果比較時有顯著差異可能表明您的模型可能存在公平性問題。

公平指標實作
------

本篇文章採用 TensorFlow 官方 [Colab 範例 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb#scrollTo=GRIjYftvuc7b) 使用 ( [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ) 資料集訓練一個簡單的神經網絡模型來檢測出微笑的明星圖像，部分說明以動畫呈現幫助您理解操作過程。

-   使用公平指標 `fairness-indicators` ，根據跨年齡組的常用公平指標評估模型性能。
-   設置一個簡單的約束優化問題，以在各個年齡段實現更公平的性能。
-   重新訓練現在"受約束的"模型並再次評估性能，確保我們選擇的公平性指標得到改善。

### 1\. 前置作業

-   本示範主要以`fairness-indicators` 、 `tensorflow_constrained_optimization` (TFCO) 模組與相依套件進行示範， `fairness-indicators` 為 Google 開源的模組，可以比較模型分類結果的公平指標，依賴 TensorFlow Extended (TFX) 的模型分析 TFMA 模組。 `fairness-indicators` 可以在驗證資料與模型分析時呈現，另外也可以運用在 TensorBoard 。
    
    ```python
    !pip install -q -U pip==20.2
    
    !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
    !pip install -q tensorflow-datasets tensorflow
    !pip install fairness-indicators \
      "absl-py==0.12.0" \
      "apache-beam<3,>=2.31" \
      "avro-python3==1.9.1" \
      "pyzmq==17.0.0"
    
    ```
    
    -   在 Colab 安裝完需重新啟動執行階段 (Restart Runtime)。
-   下載 CelebA 資料集
    
    -   [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 是一個擁有超過 20 萬張名人圖像的大規模人臉屬性數據集，每個圖像有 40 個屬性註釋（如頭髮類型、時尚配飾、面部特徵等）和 5 個標誌性位置（眼睛、嘴巴和鼻子位置）。
    -   建立 "young" 標籤作為預測結果的 y，是布林值。
    -   將圖片轉為28X28像素方便訓練。
-   建立訓練模型
    
    -   以`tf.keras.Sequential`建立基本模型。
    -   設定輔助函數、預處理函數、 用來使用 TFMA 的評述設定等。
-   檢視資料
    
    -   看明星們心情很好，但請記得此資料集不能作為商業用途喔。  
        ![](https://i.imgur.com/4Vk8lKm.png)
    -   用`df.info()`得知除了標示位置有座標、影像為矩陣之外，其餘特徵都以布林值表示。  
        ![](https://i.imgur.com/EIiHipd.png)

### 2\. 訓練與評估模型

-   在測試數據上評估模型的最終準確度得分應略高於 85%。
-   然而，跨年齡組評估的表現可能會揭示一些缺點。為了進一步探索這一點，我們使用公平指標（通過 TFMA）評估模型。我們特別感興趣的是，在評估誤報率時，"young" 的二元分類之間的性能是否存在顯著差異。
-   當名人“不微笑”的圖像並且模型預測“微笑”時，就會出現假陽性 (FP) 結果。我們可以透 FPR 作為測試準確性的衡量標準。雖然在這種情況下這是一個相對普通的錯誤，但誤報錯誤有時會導致更多的問題行為，例如垃圾郵件分類器中的誤報錯誤可能會導致用戶錯過重要電子郵件。
-   您可以透過`tfma.addons.fairness.view.widget_view.render_fairness_indicator(eval_results_unconstrained)` 指令視覺化查看是否年輕/微笑的 FPR 情形。  
    ![](https://i.imgur.com/X7tB4Xj.gif)

### 3\. 約束模型設置

-   使用[TFCO](https://github.com/google-research/tensorflow_constrained_optimization/blob/master/README.md) ，幫助限制問題：
    1.  `tfco.rate_context()` 用於構建約束。
    2.  `tfco.RateMinimizationProblem()`設定年齡類別小於或等於 5% 的誤報率將被設置為約束。
    3.  `tfco.ProxyLagrangianOptimizerV2()` – 這是真正解決速率約束問題的幫手。
        
        ```python
        # 摘述TVCO相關部分內容
        # 創建整份內容，子內容設定groups_tensor < 1，即"not young"的圖片。
        context = tfco.rate_context(predictions, labels=lambda:labels_tensor)
        context_subset = context.subset(lambda:groups_tensor < 1)
        
        # 約束設定為 FPR 小於等於 0.05 。
        constraints = [tfco.false_positive_rate(context_subset) <= 0.05]
        
        # 設置最小化錯誤
        problem = tfco.RateMinimizationProblem(tfco.error_rate(context), constraints)
        
        # 建立約束優化器，
        optimizer = tfco.ProxyLagrangianOptimizerV2(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              constraint_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              num_constraints=problem.num_constraints)
        
        # 取得使用優化器 TVCO 取得所有可訓練變量list
        var_list = (model_constrained.trainable_weights + list(problem.trainable_variables) +
                    optimizer.trainable_variables())
        
        ```
        
-   該模型現已建立並準備好使用跨年齡組的誤報率(偽陽率, FPR) 約束進行訓練。
-   TFCO 模組功能包含 `tfco.find_best_candidate_index()` 可以幫助從每個 epoch 中選擇最佳模型。
-   `tfco.find_best_candidate_index()` 可以視為一種附加的啟發式方法，它根據訓練數據的準確性和公平性約束（在本例中為跨年齡組的 FPR ）對每個結果進行排名。這樣，它可以在整體準確性和公平性約束之間尋找更好的權衡。
-   我們可以使用 `false_positive_rate` 以查看我們感興趣的指標，經過 TFCO 約束優化後，對於不年輕的子資料集，假陽率 FPR 從 0.077 下降為 0.12 ，成功減緩因為年齡造成的預測錯誤問題。
-   ![](https://i.imgur.com/OzQNb4l.png)

小結
--

-   由於 TFCO 能夠幫助該模型實現了更理想的結果，能夠找到一個接近滿足約束並儘可能減少分組之間差異的模型。實現公平的可能，但仍有改進空間，改進的方向除了透過數據 FPR 觀察，也應該有意識的探索公平性主題。
-   探索公平性主題可以求助領域專家，您也可以針對所有特徵進行切片檢視 FPR ，設定需要關注的門檻值，如果時間允許可以更深入的探究誤報問題，也請留意探索公平性是更廣泛的評估 UX 的一部分。
-   鑒於人文如此複雜，一個好的經驗法則是盡量多對資料切片觀察，留意可能較敏感的種族、性別、宗教等議題。

![/images/emoticon/emoticon07.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon07.gif)

參考
--

-   [就業服務法第5條](https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=N0090001&flno=5)
-   [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide)
