# [Day 27 : 使用 TensorFlow Serving 部署 REST API](https://ithelp.ithome.com.tw/articles/10272257)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10272257)

-   在網路情境常以 API 請求服務，用於生產的機械學習亦可用 REST API 形式提供服務。在[Day 20](https://ithelp.ithome.com.tw/articles/10267328)、[Day 21](https://ithelp.ithome.com.tw/articles/10268124)、[Day 22](https://ithelp.ithome.com.tw/articles/10268783) 介紹部署在算力有限的終端設備可採用的 TensorFlow Lite，本篇將使用 透過 TensorFlow Serving 部署您的模型作為網路服務可用的 REST API。
-   範例參考[官方範例](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)調整而成，[Colab ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1-9cCg9xhWQb7itAcLhko8UqpNOFnu-nx) 。

使用 TensorFlow Serving 部署 REST API 實作
------------------------------------

-   在此範例中，您將部署一個具有分類能力的 REST API 服務，流程為
    -   建立與訓練模型。
    -   安裝與啟動 TensorFlow Serving。
    -   以 Requests 向 TensorFlow Serving 提出請求。

### 1\. 下載資料及訓練模型

-   由於重點在如何啟動一個TensorFlow Severing 服務，資料採用 `keras.datasets.cifar10` 進行示範，[CIFAR10](https://keras.io/api/datasets/cifar10/) 為小型的影像分類資料集，具有 50,000 筆訓練資料集與 10,000 筆測試資料集，皆為 32X32 像素的彩色圖片。更多資訊參閱[官方介紹](https://www.cs.toronto.edu/~kriz/cifar.html)， 10 種分類與描述如下:  
    ![](https://i.imgur.com/PrFLRq6.png)
    
-   建立簡易模型，程式碼參閱 [Colab](https://colab.research.google.com/drive/1-9cCg9xhWQb7itAcLhko8UqpNOFnu-nx) 。  
    ![](https://i.imgur.com/SR4HB0a.png)
    

### 2\. 儲存模型

-   將模型保存為[SavedModel](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model)格式，以便將模型加載到 TensorFlow Serving 中。
    
-   [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)允許我們在發出推理請求時選擇要使用的模型版本或「可服務」版本。每個版本將導出到給定路徑下的不同子目錄。為此，需在目錄創建 `protobuf` 文件，並將包含一個版本號，此版本號碼也是後續 API 所使用的參數之一。
    
-   以下會在`/tmp/`建立版次版次`version = 1`之相關檔案。
    
    ```python
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors,
    # and stored with the default serving key
    import tempfile
    
    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print(f'export_path = {export_path}')
    
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    
    print('Saved model:')
    !ls -l {export_path}
    
    ```
    
    ![](https://i.imgur.com/dz0pKwR.png)
    

### 3\. 檢查我們的Saved model

-   `saved_model_cli` 可以檢查前述[SavedModel](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/MetaGraphDef)中相關資訊，這對理解模型相當有用，包含:
    -   [MetaGraphDefs](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/MetaGraphDef)（模型）
    -   [SignatureDefs](https://www.tensorflow.org/tfx/tutorials/signature_defs)（您可以調用的方法）
-   SavedModel CLI的詳細說明可參閱[TensorFlow 指南](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel)。
    
    ```python
    !saved_model_cli show --dir {export_path} --all
    
    ```
    
    ![](https://i.imgur.com/ErI7EPu.png)  
    (以下略)

### 4\. 建立 TensorFlow Serving 服務

-   依官方範例此為Colab環境所需設定內容，如使用本機端的 Notebook ，請注意相關提醒。
    
-   使用 [Aptitude](https://wiki.debian.org/Aptitude) 安裝 TensorFlow Serving，因為 Colab 在 Debian 環境中運行，所以可以安裝 Debian 相關套件。將 `tensorflow-model-server` 以 Root 身分添加到 Aptitude 知道的套件列表中。
    
-   另外最簡單的方式是以 Docker 佈署 TensorFlow Serving，您可自行參考[Docker 範例](https://www.tensorflow.org/tfx/serving/docker)。
    
    ```python
    
    !echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | {SUDO_IF_NEEDED} tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | {SUDO_IF_NEEDED} apt-key add -
    !{SUDO_IF_NEEDED} apt update
    
    !{SUDO_IF_NEEDED} apt-get install tensorflow-model-server
    
    ```
    

### 5\. 啟動 TensorFlow Serving

-   加載後，我們可以開始使用 REST 發出推理請求，相關參數:
    
    -   `rest_api_port`： REST 請求的 Port。
    -   `model_name`：您將在 REST 請求的 URL 中使用它。
    -   `model_base_path`：保存模型的目錄的路徑。
    
    ```python
    %%bash --bg 
    nohup tensorflow_model_server \
      --rest_api_port=8501 \
      --model_name=fashion_model \
      --model_base_path="${MODEL_DIR}" >server.log 2>&1
    
    
    ```
    

### 6\. 向 TensorFlow Serving 提出請求

-   預設請求最新版本的 servable 。
    
    ```python
    import requests
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://localhost:8501/v1/models/fashion_model:predict', 
        data=data, 
        headers=headers
        )
    
    predictions = json.loads(json_response.text)['predictions']
    
    show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
      class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[int(test_labels[0])], test_labels[0]))
    
    ```
    
-   向伺服器請求指定版本`version = 1` 。
    
    ```python
    # docs_infra: no_execute
    version = 1
    
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        f'http://localhost:8501/v1/models/fashion_model/versions/{version}:predict', 
        data=data, 
        headers=headers
        )
    
    predictions = json.loads(json_response.text)['predictions']
    
    for i in range(0,3):
      show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
        class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[int(test_labels[i])], test_labels[i]))
    
    
    ```
    
    ![](https://i.imgur.com/71hvMBT.png)
    

小結
--

-   由於提供網路服務的工程師相當熟悉 REST API 的溝通方式，以指定 URI 即可取得所需資訊相當便捷，您也會發現 TensorFlow Serving 簡化諸多部署 API 的設定細節。
-   截至此篇文章將實踐 MLOps 在範疇、資料、模型與部署的相關概念與做法介紹後，接連 2 篇將接紹用於生產的機械學習框架 TensorFlow Extended (TFX) ，謝謝您一路堅持至今，我們下篇見。  
    ![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif)

參考
--

-   [TensorFlow Serving 範例說明](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
