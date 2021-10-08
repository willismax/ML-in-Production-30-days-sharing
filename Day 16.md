# [Day 16 : 特徵工程 tf.Tramsform 實作](https://ithelp.ithome.com.tw/articles/10264084)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10264084)


接續 [Day 15](https://ithelp.ithome.com.tw/articles/10263595) 的 tf.Tramsform 介紹，今日進行實作，先以[TensorFlow Transform 預處理數據的入門範例](https://www.tensorflow.org/tfx/tutorials/transform/simple) 作為演示過程，[官方 Colab 支援 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/transform/simple.ipynb)， 您會發現 Apache Beam 的 pipeline 撰寫方式融入在程式中。

TFT 特徵工程簡易入門
------------

### 1\. 建立環境

-   因為 Colab 的套件版本問題，pip 需更新並安裝 tensorflow_transform 套件，安裝需要時間，並且應該要重新啟動(執行階段>重新啟動執行階段)。
    
    ```python
    try:
      import colab
      !pip install --upgrade pip
    except:
      pass
    
    !pip install -q -U tensorflow_transform==0.24.1
    
    ```
    
-   引入的模組不少，分別介紹:
    
    -   `pprint` 輸出比較漂亮。
    -   `tempfile` 生成暫存檔案所需。
    -   `tensorflow` 已經是2.x版需引入。
    -   `tensorflow_transform` 簡稱 `tft` 。
    -   `tensorflow_transform.beam` 實現使用 Apache Beam 。
    -   `tensorflow_transform.tf_metadata` 為紀錄數據所需要的 metadata 模組，引入 `dataset_metadata` 及 `schema_utils`。
    
    ```python
    import pprint
    import tempfile
    
    import tensorflow as tf
    import tensorflow_transform as tft
    
    import tensorflow_transform.beam as tft_beam
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import schema_utils
    
    ```
    

### 2\. 創建資料及中繼資料 matadata

-   在此範例的模擬假資料，您就當是 tft 的 Hello world 。
-   用於生產的機械學習基於任務情境及後續應用，接收到的資料很有可能是採用 JSON 形式，透過 Request 取得的 RESTFUL 資料。
    
    ```python
    raw_data = [
          {'x': 1, 'y': 1, 's': 'hello'},
          {'x': 2, 'y': 2, 's': 'world'},
          {'x': 3, 'y': 3, 's': 'hello'}
      ]
    
    ```
    
-   定義資料特徵的 Schema 。
    
    ```python
    raw_data_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec({
            'y': tf.io.FixedLenFeature([], tf.float32),
            'x': tf.io.FixedLenFeature([], tf.float32),
            's': tf.io.FixedLenFeature([], tf.string),
        }))
    
    ```
    

### 3\. 撰寫預處理函數 preprocessing_fn()

-   tf.Tramsform 已經實作許多特徵工程的函數，您可以參考 [TFX API 文件](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)。
    
    ```python
    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        x = inputs['x']
        y = inputs['y']
        s = inputs['s']
        x_centered = x - tft.mean(x)
        y_normalized = tft.scale_to_0_1(y)
        s_integerized = tft.compute_and_apply_vocabulary(s)
        x_centered_times_y_normalized = (x_centered * y_normalized)
        return {
            'x_centered': x_centered,
            'y_normalized': y_normalized,
            's_integerized': s_integerized,
            'x_centered_times_y_normalized': x_centered_times_y_normalized,
        }
    
    ```
    

### 4\. 組合流程

-   現在我們已準備好轉換我們的數據。我們將使用帶有直接運行器的 Apache Beam，輸入為：
    
    -   `raw_data` : 我們上面創建的原始輸入數據。
    -   `raw_data_metadata` : 原始數據的 Schema。
    -   `preprocessing_fn` : 預處理的特徵工程函數。
-   關於 Apache Beam 的特殊語法用到了會在 Linux 使用的 `|` (pipe 運算子)，可以理解為:
    
    -   `=` 左邊是最終結果 result。
    -   `=` 右邊第一個是輸入 pass_this，接續 `|` 是執行過程步驟。
    -   下方每個 `|` 右邊先`命名`，再`>>`執行。也可以省略命名直接 to\_this\_call。
    
    ```
    result = pass_this | 'name this step' >> to_this_call
    
    result = apache_beam.Pipeline() | 'first step' >> do_this_first() | 'second step' >> do_this_last()
    
    ```
    
-   在此範例，創建了暫時資料夾，將`raw_data, raw_data_metadata`作為 `tft_beam.AnalyzeAndTransformDataset( preprocessing_fn)` 的輸入，執行結果輸出存入 `transformed_dataset, transform_fn` 。
    
-   最後將 `transformed_dataset` 拆成 `transformed_data` , `transformed_metadata` ，並列印原始資料以及經過前處理的資料對照。
    
    ```python
    def main():
      # Ignore the warnings
      with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (  
            (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn))
    
      transformed_data, transformed_metadata = transformed_dataset  
    
      print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
      print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))
    
    if __name__ == '__main__':
      main()
    
    ```
    
    ![](https://i.imgur.com/PMsVKvH.png)
    

小結
--

-   以上為TensorFlow Transform 預處理數據的入門範例演示過程，高級版的已經包含後續訓練與評估的流程，您可以逕行參考執行。
-   筆者理解 tf.Transform 花了不少時間，這篇簡要整理您所需之道的內容，仍希望能幫助到您。

參考
--

-   [TensorFlow Transform 預處理數據的入門範例](https://www.tensorflow.org/tfx/tutorials/transform/simple)
-   [Apache Beam](https://beam.apache.org/)
