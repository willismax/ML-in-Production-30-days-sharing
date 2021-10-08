# [Day 18 : 深度學習(神經網絡)自動調參術 - KerasTuner](https://ithelp.ithome.com.tw/articles/10265801)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10265801)


-   接續將關注焦點來到 Model 的主題，在您閱讀本系列文章之前，您或許已有建模經驗，在用於生產的機械學習情境，手動調參優化模型與資料是耗費人時的吃重工作，自動化訓練、調參機制成為無可避免的選擇，讓您將時間投入在更需要在意的問題之中。
-   在本系列 Model 的主題，將介紹如何自動化選擇與訓練模型，以及優化模型的有趣技巧。這篇說明的是自動化調整超參數的 KerasTuner。
-   Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1qkot7-OLWTf9F5SyaS0_6doBKCy33qM5?usp=drive_fs)。

Hyperparameter turning 超參數調參
----------------------------

-   在您的 ML 系統中，輸入可歸納為「資料」、「超參數」及「模型」，在機械學習領域中，能否選擇良好的超參數，通常是決定機器學習專案成敗的關鍵，對於更複雜的模型，超參數的數量會急劇增加，手動調整它們可能非常具有挑戰性。有沒有可能用自動化的方式搜尋最佳參數?
-   超參數類型有兩種：
    -   Model hyperparameters:
        -   影響模型選擇的模型超參數，例如隱藏層的數量和寬度。
    -   Algorithm hyperparameters:
        -   影響學習的演算法，例如 SGD 的學習率和 KNN 的 K 值。

KerasTurner
-----------

-   [Keras Tuner](https://keras.io/keras_tuner/)是 Keras 團隊的一個模組，可自動執行神經網絡的超參數調整
-   為了進行比較，首先使用預先選擇的超參數訓練 Baseline 模型，然後使用調整後的超參數重做該過程。
-   範例改寫自[Tensorflow 提供的官方教學](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/keras_tuner.ipynb#scrollTo=sKwLOzKpFGAj)，採用採用 [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) 資料集。提供您 Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1qkot7-OLWTf9F5SyaS0_6doBKCy33qM5?usp=drive_fs)。

### 建立基準模型

-   先定義基本模型作為優化基準。  
    ![](https://i.imgur.com/goT31ee.png)
-   為了方便顯示結果，範例有定義一個輔助函數。  
    ![](https://i.imgur.com/j6bTCRK.png)

### 定義模型

-   當您構建調整超參數的模型時，除了模型架構之外，您還定義了超參數搜索空間。為了調整超參數的模型稱為 Hyper Model。
    
-   您可以通過兩種方法定義 Hyper Model：
    
    -   通過使用模型構建器功能(本範例採用)。
    -   通過`HyperModel`繼承 `Keras Tuner`類別。
-   另外兩個預定義的HyperModel類別 [`HyperXception`](https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperxception-class)和[`HyperResNet`](https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperresnet-class)可用於計算機視覺應用程序。
    
-   以下模型函數中，
    
    -   `Int()`用來定義密集單元的搜索空間的最小值和最大值。
    -   `Choice()`用於設定學習率。
    
    ```python
    def model_builder(hp):
    
      model = keras.Sequential()
      model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
      # 設定搜索值範圍
      hp_units = hp.Int('units', min_value=32, max_value=512, step=32) 
    
      model.add(
          keras.layers.Dense(
              units=hp_units, 
              activation='relu', 
              name='dense_1'
              )
          )
    
      model.add(keras.layers.Dropout(0.2))
      model.add(keras.layers.Dense(10, activation='softmax'))
    
      # 設定學習率範圍 0.01, 0.001, or 0.0001
      hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
      model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
          loss=keras.losses.SparseCategoricalCrossentropy(),
          metrics=['accuracy']
          )
      return model
    
    ```
    

### 實例化 Tuner 並執行超調

-   在此設定 Tuner 如下，細節調整您可以參考[官方文件](https://keras.io/api/keras_tuner/tuners/hyperband/)。
    
    ```python
    # Instantiate the tuner
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='kt_dir',
        project_name='kt_hyperband'
        )
    
    ```
    
-   另外為了節省不必要的訓練時間，定義了一個 [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)，當驗證的 loss 在 5 個 epoch 沒改善時停止訓練，以下為持續搜尋的過程。  
    ![](https://i.imgur.com/aG1Lszn.png)
    
-   最後將產生一組最佳超參數搜尋結果。
    

### 以最佳超參數訓練模型並比對基準

-   範例搜尋到最佳的超參數為隱藏層設定 480 個神經元，學習率為 0.001。  
    ![](https://i.imgur.com/akTH70c.png)
-   經比對有自動搜尋出神經元略低、準確度略高、loss略低的超參數調整結果，雖然不明顯。  
    ![](https://i.imgur.com/rpnX9Vx.png)

小結
--

-   在本篇您使用 Keras Tuner 方便地調整超參數。您定義了要調整的參數、搜索空間和搜索策略，以達到最佳超參數集。雖然要搜尋最佳參數過程消耗運算資源，但可讓您騰出時間做更重要的事。
-   用於生產的機械學習情境，您會有因為時間推移產生的資料偏移、概念篇移、模型效果衰退問題，觸發自動化持續訓練機制，您可以建構較可靠的機械學習服務。

參考
--

-   [Keras Tuner](https://keras.io/keras_tuner/)
-   [Tensorflow 提供的官方教學](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/keras_tuner.ipynb#scrollTo=sKwLOzKpFGAj)
