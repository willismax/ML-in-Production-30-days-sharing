# [Day 21 : 模型優化 - 剪枝 Pruning](https://ithelp.ithome.com.tw/articles/10268124)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10267328)

-   如果說可以讓模型縮小10倍，精度還維持水準，這是什麼巫術?
-   延續 [Day 20](https://ithelp.ithome.com.tw/articles/10267328) 的模型優化作法，本次再結合剪枝技術做到更輕量的模型效果。

什麼是剪枝 Pruning
-------------

-   [剪枝 Pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras) 將無關緊要的權重 (weight) 刪除歸零，在壓縮時因為稀疏矩陣的特性，能明顯縮小尺寸，可以壓縮到原來 1/3。
-   如果經過剪枝再量化的模型，甚至可以縮小的原來 1/10 大小。

模型優化剪枝實作
--------

-   Colab 支援 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1QQ0rZ9f18APlBy23M3-hfTMb4a5LHFtw)。
    
-   本示範採用 Tensorflow 模型優化模組的 `prune_low_magnitude()` ，可以將 Keras 模型在訓練期間將影響較小的權重修剪歸零。
    
    ```python
    !pip install tensorflow\_model\_optimization
    
    ```
    

### 建立基本模型

-   我們的基本模型以[訓練後量化](https://colab.research.google.com/drive/1ukgVrMdtWjpReIygWHJ7-Lcw61Lv5kAO) 相同的基準模型進行優化，模型一樣採用`tf.keras.datasets.mnist`，用CNN進行建模。
    
    ![](https://i.imgur.com/JARv0UH.png)
    
    ```python
    ACCURACY:
    {'baseline Keras model': 0.9574999809265137}
    
    MODEL_SIZE:
    {'baseline h5': 98136} 
    
    ```
    

### 使用剪枝調整模型

-   進行剪枝，另外因為剪枝模型方法有增加一層包裝層，摘要顯示的參數會增加。
    
    ```python
    # Get the pruning method
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1
    
    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    
    # Define pruning schedule.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50,
            final_sparsity=0.80,
            begin_step=0,
            end_step=end_step)
        }
    
    # Pass in the trained baseline model
    model_for_pruning = prune_low_magnitude(
        baseline_model, 
        **pruning_params
        )
    
    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    
    model_for_pruning.summary()
    
    ```
    
    ![](https://i.imgur.com/gNUcYom.png)
    
-   參數數量增加了。
    

### 觀察剪枝前後的模型權重 weight 變化

-   剪枝前，有些微弱的權重。  
    ![](https://i.imgur.com/QTWCL3H.png)
    
-   重新訓練模型。並在 Callback 增加 `tfmot.sparsity.keras.UpdatePruningStep()` 參數。
    
    ```python
    # Callback to update pruning wrappers at each step
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    
    # Train and prune the model
    model_for_pruning.fit(
        train_images, 
        train_labels,
        epochs=epochs, 
        validation_split=validation_split,
        callbacks=callbacks
        )
    
    ```
    
-   重新訓練後已修剪，觀察同一層的權重變化，許多不重要的權重已歸零。  
    ![](https://i.imgur.com/7v40hXw.png)
    

### 剪枝後移除包裝層

-   剪枝之後，您可以用tfmot.sparsity.keras.strip_pruning()刪除包裝層以具有與基線模型相同的層和參數。
    
-   此方法也有助於保存模型並導出為*.tflite檔案格式。  
    ![](https://i.imgur.com/zRriN8g.png)
    
-   剪枝後尚未壓縮的檔案，模型檔案大小與原先一致，這也挺合理的畢竟都還占著位子。
    
    ```python
    MODEL_SIZE:
    {'baseline h5': 98136,
    'pruned non quantized h5': 98136} 
    
    ```
    

### 模型壓縮3倍術

-   剪枝後的模型再壓縮。
    
-   壓縮後檔案大小約為原本1/3，這是因為剪枝後歸零的權重可以更有效的壓縮。
    
    ```python
    import tempfile
    import zipfile
    
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write('pruned_model.h5')
    
    
    MODEL_SIZE['pruned non quantized h5'] = get_gzipped_model_size('pruned_model.h5')
    
    ```
    
    ```python
    MODEL_SIZE:
    {'baseline h5': 98136,
    'pruned non quantized h5': 25665} 
    
    ```
    

### 模型壓縮10倍術

-   現在嘗試將已精剪枝後的模型再量化。
    
-   量化原本就會縮小約3倍，將剪枝模型壓縮後再量化，與基本模型相比，這使模型大小減少了約為原本1/10，而且精度還能維持水準。
    
    ```python
    # 剪枝壓縮後再量化模型
    converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open('pruned_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    ```
    
    ```python
    MODEL_SIZE:
    {'baseline h5': 98136,
     'pruned non quantized h5': 25665,
     'pruned quantized tflite': 8129}
    
     ACCURACY
     {'baseline Keras model': 0.9574999809265137,
     'pruned and quantized tflite': 0.9683,
     'pruned model h5': 0.9685999751091003}
    ```
    

小結
--

-   本篇示範減少模型檔案大小: 剪枝、壓縮再量化，原本的 `.h5` 檔案轉換為 TensorFlow Lite 的 `*.tflite` 檔案可以是原本的 1/10 ，相當神奇，也推薦給有需要的您。

![/images/emoticon/emoticon12.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon12.gif)

參考
--

-   [TensorFlow Lite官方範例](https://www.tensorflow.org/lite/performance/post_training_quantization)。
-   [Pruning Comprehensive Guide](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)
-   啟發於 [Machine Learning Modeling Pipelines in Production](https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production) 課程。
