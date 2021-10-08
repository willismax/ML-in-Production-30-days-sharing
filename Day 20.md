# [Day 20 : 模型優化 - 訓練後量化 Post Training Quantization](https://ithelp.ithome.com.tw/articles/10267328)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10267328)

-   當我們訓練模型需要部署在硬體較為受限的智慧型裝置、IOT設備，模型運算在吃緊的硬體資源中顯得笨重，此時可以採取模型優化策略改進。
-   量化 Quantization 的好處:
    -   神經網路的參數真的很多也需要空間。
    -   減少檔案大小。
    -   減少運算資源。
    -   達到更快更輕小的優化成果。

什麼是訓練後量化 Post Training Quantization
-----------------------------------

-   訓練後量化 Post Training Quantization 是一種轉換技術，可以減少模型大小，同時還可以改善 CPU 和硬件加速器的延遲，模型精度幾乎沒有下降。
-   有多種訓練後量化選項可供選擇。以下是選擇及其提供的好處的匯總表：

| 技術 | 好處 | 硬體 |
| --- | --- | --- |
| 動態範圍量化 | 小 4 倍，加速 2x-3x | CPU |
| 全INT量化 | 小 4 倍，加速 3x+ | CPU、Edge TPU、微控制器 |
| Float16 量化 | 小 2 倍，GPU 加速 | CPU、GPU |

-   各種量化技術使用需求，TensorFlow Lite 文件整理出您可以透過以下決策樹協助判斷解決方案，幫助確定哪種訓練後量化方法最適合您的用例：  
    ![](https://i.imgur.com/nQwPqQ8.png)
    -   圖片來源: [TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quantization) 。

如何進行訓練後量化 Post Training Quantization
------------------------------------

-   當您使用TensorFlow Lite Converter將已訓練的 TensorFlow 模型轉換為 TensorFlow Lite 格式時，您可以對其進行量化 。
    
-   另外 Pytorch 也有 [QUANTIZATION](https://pytorch.org/docs/stable/quantization.html) 實作。
    
-   以下實作範例可用 Colab 執行，另請注意 TensorFlow 版本需 >= 1.15 。
    
-   Colab 實作 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1ukgVrMdtWjpReIygWHJ7-Lcw61Lv5kAO)
    

### 建立基本模型

-   模型採用`tf.keras.datasets.mnist`，用CNN進行建模。
-   過程中儲存基本模型權重檔案`baseline_weights.h5`，儲存量化前的模型`non_quantized.h5`，並記錄模型大小與準確率，以進行訓練後量化的比較。
-   模型基本架構:  
    ![](https://i.imgur.com/xlGeUm3.png)

### 轉為 TF Lite 格式

-   TensorFlow Lite 使用 `*.tflite`格式，用`tf.lite.TFLiteConverter.from_keras_model`轉換先前建立的baseline_model。
    
    ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
    
    tflite_model = converter.convert()
    
    with open('non_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    ```
    
-   建立TF Lite 的評估模型準確率的函數，轉檔為`tflite`後需要特別撰寫評估函數，參考並改寫[官方範例](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8#evaluate_the_models)。
    
    ```python
    # A helper function to evaluate the TF Lite model using "test" dataset.
    # from: https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8#evaluate_the_models
    def evaluate_model(filemane):
      #Load the model into the interpreters
      interpreter = tf.lite.Interpreter(model_path=str(filemane))
      interpreter.allocate_tensors()
    
      input_index = interpreter.get_input_details()[0]["index"]
      output_index = interpreter.get_output_details()[0]["index"]
    
      # Run predictions on every image in the "test" dataset.
      prediction_digits = []
      for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
    
        # Run inference.
        interpreter.invoke()
    
        # Post-processing: remove batch dimension and find the digit 
        # with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
    
      # Compare prediction results with ground truth labels to calculate accuracy.
      accurate_count = 0
      for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
          accurate_count += 1
      accuracy = accurate_count * 1.0 / len(prediction_digits)
    
      return accuracy
    
    ```
    
-   此時評估模型的準確率相近，模型尺寸減少。
    
    ```python
    ACCURACY:
    {'baseline Keras model': 0.9581000208854675, 
     'non quantized tflite': 0.9581}
    
    MODEL_SIZE:
    {'baseline h5': 98136, 
     'non quantized tflite': 84688}
    
    ```
    

### 訓練後量化 Post-Training Quantization

-   本範例示範訓練後量化之動態範圍量化 Dynamic range quantization ，您也可以嘗試固定float8、float16量化。
    
    ```python
    # Dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] #增加此設定
    tflite_model = converter.convert()
    
    with open('post_training_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    ```
    
-   模型大小下降為原來1/4，精準度相近。
    
    ```python
    ACCURACY:
    {'baseline Keras model': 0.9581000208854675,
     'non quantized tflite': 0.9581,
     'post training quantized tflite': 0.9582}
    
    MODEL_SIZE:
    {'baseline h5': 98136,
     'non quantized tflite': 84688,
     'post training quantized tflite': 24096} 
    
    ```
    

### (選用)量化感知訓練 Quantization Aware Training

-   當訓練後量化導致您的準確率下降多到無法接受，可以考慮在量化模型之前進行[量化感知訓練 Quantization Aware Training](https://www.tensorflow.org/model_optimization/guide/quantization/training)。
-   此方法為在訓練期間在模型中插入假量化節點來模擬精度損失，讓模型學會適應精度損失，以獲得更準確的預測。
-   需額外安裝 `tensorflow_model_optimization` 模組，該模組提供 `quantize_model()` 完成任務。
-   在 Colab 的示範中，是使用先前初步訓練的 'baseline_weights.h5' 模型權重進行優化。您會發現模型增加了些假結點與 Layer。另訓練經過感知訓練的模型，您可以自行調整 epochs，範例只用 `epochs = 1`。  
    ![](https://i.imgur.com/fVh8WTv.png)
-   先感知訓練後，模型經度略為改變，尺寸略增。
    
    ```python
    ACCURACY:
    {'baseline Keras model': 0.9581000208854675,
     'non quantized tflite': 0.9581,
     'post training quantized tflite': 0.9582,
     'quantization aware non-quantized': 0.1005999967455864}
    
    MODEL_SIZE:
    {'baseline h5': 98136,
     'non quantized tflite': 84688,
     'post training quantized tflite': 24096,
     'quantization aware non-quantized': 115680} 
    
    ```
    
-   調整後再執行 Post-Training Quantization 可舒緩準確率下降的問題，但本案例沒有明顯的精度損失，您有需要再試即可。

小結
--

-   透過 Post Training Quantization 可以很明顯的發現檔案比單純轉換為 TensorFlow Lite 檔案更小，會是未轉換前的1/4，對IOT設備壓力減輕很多，也是您可以採用的優化方案。本篇也提供了可以減緩轉換過程精度損失過大的優化方式，供您參考。
-   明日也會再與您分享與實作另一種優化方式-剪枝，我們下回見。

參考
--

-   [Post Training Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)
-   [Quantization Aware Training Comprehensive Guide](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide)
