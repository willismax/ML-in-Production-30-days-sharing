# [Day 17 : 用於生產的機械學習 - 特徵選擇 Feature Selection](https://ithelp.ithome.com.tw/articles/10264846)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10264846)


特徵選擇是機器學習中的核心概念之一，不相關或部分相關的特徵會對模型性能產生負面影響，也會有效能的問題，適當的挑選與目標變量最相關的特徵集，有助降低模型的複雜性，並最大限度地減少訓練和推理所需的資源。這在您可能處理 TB 級數據或服務數百萬個請求的生產模型中具有更大的影響，以下說明幾個特徵選擇的作法。

-   Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1Y37iCwCCaaSg8U-mHrETtqWf3IPn2b9V?usp=drive_fs)。

特徵選擇方法
------

-   非監督式學習
    -   可以採取移除關聯較低的特徵策略，減少無關的特徵。
-   監督式學習
    -   關聯度高的特徵通常有重疊性，可以找出與之關聯性高的其他特徵，保留部分特徵並移除原先觀察之特徵。

監督式學習適用的特徵選擇
------------

-   過濾方法 Filter Method
    -   關聯分析 Correlation。
    -   單變量特徵選取 Univariate feature selection
-   包裝方法 Wrapper Method
    -   前向消除 Forward elimination
    -   後向消除 Backward elimination
    -   遞迴特徵消除 Recursive feature elimination (RFE)
-   嵌入方法 Embedded Method
    -   重要特徵 Feature importance
    -   L1正規化 L1 regularization

特徵選擇示範，鐵達尼存活資料集為例
-----------------

-   資料集為經整理過後的[鐵達尼號資料集](https://raw.githubusercontent.com/duxuhao/Feature-Selection/master/example/titanic/clean_train.csv)，Model主要以`sklearn.ensemble.RandomForestClassifier`進行示範，程式碼參考 [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) 課程，經過改並以不同資料及呈現過程，執行細節請參見 Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1Y37iCwCCaaSg8U-mHrETtqWf3IPn2b9V?usp=drive_fs)。

### 0\. 前置處理

-   整理特徵 X ，部分屬於分類資料可以用 One-Hot Encoding編碼，並且處理無效欄位。  
    ![](https://i.imgur.com/Cy09eTA.png)
    
-   定義評估模型，固定以 `RandomForestClassifier` 模型進行訓練，並對照各測試資料集之績效。
    
    ```python
    def use_RandomForestClassifier_evaluation_metrics_on_test_set(X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size = 0.2 ,stratify=Y, random_state = 9527)
    
        # 標準化
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # RandomForestClassifier訓練模型
        model = RandomForestClassifier(criterion='entropy', random_state=9527)
        model.fit(X_train_scaled, Y_train)
    
        # 預測
        y_predict_result = model.predict(X_test_scaled)
    
        # 回傳evaluation_metrics_on_test_set
        return {
            'accuracy' : accuracy_score(Y_test, y_predict_result),
            'roc' : roc_auc_score(Y_test, y_predict_result),
            'precision' : precision_score(Y_test, y_predict_result),
            'recall' : recall_score(Y_test, y_predict_result),
            'f1' : f1_score(Y_test, y_predict_result),
            'Feature Count' : len(X.columns)
        }
    
    ```
    
    ![](https://i.imgur.com/QCILRGV.png)
    
-   查看關聯性，越淺代表高度正相關，越深代表高度負相關，關聯性介於 \[1,-1\] 之間。
    
    ![](https://i.imgur.com/PArVeqQ.png)
    

### 1\. 特徵選擇 - 過濾方法 Filter Method

-   **1.1 依關聯性移除特徵**
    
    -   選擇具有與其他特徵高度相關的某特徵 ，設定閾值選取符合特徵，並移除該特徵。本次實作挑選的是 `FamilySize` ，選擇超過門檻值的有代表性的特徵(無論正負相關 >0.2 )，並移除 `FamilySize` 本身。
        
        ```python
        # 取得具有與其他部分特徵高度相關的某特徵絕對值
        cor_target = abs(cor["FamilySize"])
        
        # 選擇高度相關的特徵（閾值 = 0.2）
        relevant_features = cor_target[cor_target>0.2]
        
        # 選擇特徵名稱
        names = [index for index, value in relevant_features.iteritems()]
        
        # 刪除目標特徵
        names.remove('FamilySize')
        
        print(names)
        
        ```
        
        ![](https://i.imgur.com/NGtqdid.png)  
        ![](https://i.imgur.com/T8LFFwS.png)
        
        -   對比未篩選的成績，特徵減少至9個，多數指標分數下降，recall維持不變。
-   **1.2 單變量特徵選取 Univariate Selection**
    
    -   以 `sklearn.feature_selection.SelectKBest` 選擇最具影響力的特徵，這邊示範 選取 `10` 個特徵。  
        ![](https://i.imgur.com/vBevGxR.png)

### 2\. 特徵選擇 - 包裝方法 Wrapper Method

-   **2.1 遞迴特徵消除 Recursive feature elimination (RFE)**
    -   以 `sklearn.feature_selection.RFE` 篩選，設定篩選出 `k=10` 個重要特徵，該設定也是超參數，您可以自行設定。
        
        ```python
        # Recursive Feature Elimination
        def rfe_selection( X , Y, k=10):
        
            # Split train and test sets
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, 
                Y, 
                test_size = 0.2, 
                stratify=Y, 
                random_state = 9527)
        
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
            model = RandomForestClassifier(
                criterion='entropy', 
                random_state=9527
                )
            rfe = RFE(model, k)
            rfe = rfe.fit(X_train_scaled, Y_train)
        
            feature_names = X.columns[rfe.get_support()]
        
            return feature_names
        
        ```
        
        ![](https://i.imgur.com/0hNJtwF.png)

### 3\. 特徵選擇 - 重要特徵 Feature importance

-   依重要性門檻值篩選特徵
    -   示範以 `sklearn.feature_selection.SelectFromModel` 選擇，以 `threshold=0.013` 作為篩選。  
        ![](https://i.imgur.com/h3a1rO8.png)  
        ![](https://i.imgur.com/LhJqJtM.png)

最終特徵選擇比較
--------

-   經過一系列的特徵選擇，您可以視需求選擇所要採取的特徵，您追求準確率的話「遞迴特徵消除 Recursive feature elimination (RFE) 」分數最高，原本特徵從16個減少為10個，成功減少 37.5% ，運算資源可以大幅減少，而且各項指標成績都優於全數選取的結果。  
    ![](https://i.imgur.com/Rypd2Q6.png)
    
-   您會發現過程中有許多超參數是可以自行調整的，在您的手上或許仍有優化空間，譬如將RFE的特徵改 k=9 試試看，性能與成績還可以再提升。
    

小結
--

-   特徵選擇相當有用，當在持續需要學習的情境，微小的節省資源手段都能帶來龐大的成本效益，您已經學會如何幫老闆、幫您自己省時、省錢了!
-   收藏起來以後用自己的資料集試試吧，希望能幫助到您。

參考
--

-   [鐵達尼號資料集](https://raw.githubusercontent.com/duxuhao/Feature-Selection/master/example/titanic/clean_train.csv)
-   [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production)
