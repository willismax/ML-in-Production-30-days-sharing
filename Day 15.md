# [Day 15 : 特徵工程 tf.Tramsform 介紹](https://ithelp.ithome.com.tw/articles/10263595)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10263595)

-   特徵工程是機械學習相當重要的一環，有處理數據以及實行 ML/DL 任務經驗者對特徵工程一定不陌生，一般來說常以 Pandas 及 sklearn 完成任務，也有用 Excel 打天下，能夠善用的就是好工具。
-   到了用於生產的機械學習情境，前述的工具在使用上就有限制，首先您是在生產過程 Online 進行特徵工程，資料流底層可能是 Apache Spark , Flink 等批次/串流/分散式運算情境，對於不同特徵的特徵工程計算會有時差、等待等情形，您所熟悉的模組恐怕較難因應。
-   基於前述想法，本篇介紹常用的特徵工程，並介紹用於生產中的機械學習 TFX 中的特徵工程組件 `tf.Tramsform` 。

用於生產的特徵工程 tf.Tramsform
----------------------

-   tf.Tramsform 做 TFX 的其中一個組件，主要任務為特徵工程。
    
-   tf.Tramsform 為能運用在平行運算可擴展的叢集環境，係基於 Apache Beam 的框架，您所寫的TFT特徵工程操作，雖然寫起來像 TensorFlow ，底層已轉化為用 Beam 計算。
    
-   您安排好的特徵工程通常寫在`preprocessing_fn()`自訂函式中，譬如某些特徵的資料型態轉換、資料清洗、標準化等，甚至可能複數特徵進行Bucketize、Feature Cross 等，把特徵工程的邏輯寫入，您也可以設想，現在的工作是在之後部署時面對「未知」資料流的資料工程手段。
    
-   接著我們把焦點聚焦到「訓練」與「部署」的情形，在 [TensorFlow Dev Summit 2018](https://www.youtube.com/watch?v=vdG7uKQ2eKk&feature=youtu.be&t=199) 介紹 tf.Tramsform 用於生產階段式轉化為 [tf.Graph](https://www.tensorflow.org/guide/intro_to_graphs) 的運作，轉換為tf.Graph 最大好處是可以擺脫對 Python 編譯器的依賴，可以實現 TensorFlow 並行並在多個設備上高效運行，這樣的轉換帶來訓練資料與部署時的特徵工程作業一致，也帶來在不同裝置環境一致的好處，消逆了`Training-Serving Skew`。
    
    -   ![](https://i.imgur.com/dZJX0aX.png)
        -   圖片來源:[使用 tf.Transform 對 TensorFlow 管道模式進行預處理](https://kknews.cc/code/kbvlxgv.html)
-   舉例有 X, Y, Z 等3個特徵，分別進行特徵工程，並且在分散式的運算情境中進行。
    
-   前述的運算在 tf.Tramsform 稱之為Analyze，分析出的計算套在tf.Graph 運算，兩者情境感受一致。  
    ![](https://i.imgur.com/NyHjXUL.png)
    
    -   圖片來源: [TensorFlow Dev Summit 2018](https://www.youtube.com/watch?v=vdG7uKQ2eKk&feature=youtu.be&t=199)
-   最後您可以用`tft.apply_save_model` 儲存模型。
    

### 特徵工程與 tf.Tramsform 模組常用方法整理

-   數值範圍的特徵工程
    
    -   Scaling
    -   Normalizing
    -   Standardizing
        
        ```python
        tft.scale_to_z_score
        tft.scale_0_to_1
        tft.scale_to_qaussian
        
        ```
        
-   群組化的特徵工程
    
    -   Bucketizing
        
        ```python
        tft.bucketize
        tft.quantiles
        tft.apply_buckets
        
        ```
        
-   詞彙型的特徵工程
    
    -   Bag of words
    -   TF-IDF
    -   Ngrams
        
        ```python
        tft.bag_of_words
        tft.tfidf
        tft.ngrams
        tft.string_to_int
        
        ```
        
-   降維的特徵工程
    
    -   PCA
        
        ```python
        tft.pca
        
        ```
        
-   特徵編碼的特徵工程
    
    -   One-Hot encoding
    -   Embedding
        
        ```python
        #0-N
        pandas.factorize() 
        sklearn.preprocessing.LabalEncoder() 
        
        #One-Hot
        tf.one_hot
        pandas.get_dummies()
        sklearn.preprocessing.OneHotEncoder()
        
        ```
        
-   組合特徵的特徵工程
    
    -   Feature crossing
        
        ```python
        tf.string_join
        tft.string_to_int
        
        ```
        

小結
--

-   特徵工程是機械學習的大事，本篇試圖介紹 TFX 的特徵工程組件 tf.Tramsform ，主要是因為網路上特徵不缺乏特徵工程教學，但用於生產的機械學習解決方案考慮比較嚴謹，可以看到 tf.Tramsform 為了能在生產環境運行，底層採用 Apache Beam，並且轉化為 tf.Graph 來實現高速、並行、不侷限 Python 環境的特徵工程，這在部署在伺服器、web、手機裝置有較好的適應能力。
-   後續將介紹實際運行的程式，會更清晰，我們明天見。

![/images/emoticon/emoticon07.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon07.gif)

參考
--

-   [TFX API 文件](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)
-   [https://www.commonlounge.com/discussion/3ce75d036e924c70ab7e47f534ec40fc/history](https://www.commonlounge.com/discussion/3ce75d036e924c70ab7e47f534ec40fc/history)
-   [https://www.daytime.cool/tech/4941117.html](https://www.daytime.cool/tech/4941117.html)
-   [TFX 開發峰會演講](https://www.youtube.com/watch?v=vdG7uKQ2eKk&feature=youtu.be&t=199)
-   [使用 tf.Transform 對 TensorFlow 管道模式進行預處理](https://kknews.cc/code/kbvlxgv.html)
