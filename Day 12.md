# [Day 12 : 弱監督式標註資料 Snorkel (spam 入門篇)](https://ithelp.ithome.com.tw/articles/10262325)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10262325)

-   當您需要更高效率標註大量資料時，人工標註不符合自動化的機械學習需求，採用靠著程式寫條件就分類完成的 Snokel 就可以參考。
-   而且在2021年 AI 台灣人工智慧年會，該 Snokel.ai 的 CEO 以 [The Future of Data-Centric AI](https://ithelp.ithome.com.tw/articles/(https://conf2021.aiacademy.tw/alex-ratner/)) 為題在活動第1日演講，有興趣可以關注活動。
-   Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1mgroggxRG_yuLw2OaBD8OBzDuBQ57ZTo?usp=drive_fs)。

![](https://i.imgur.com/P7o4ESK.png)

什麼是 Snorkel
-----------

-   Snorkel 是一種「無需手動標記」即可以用程式構建和管理訓練數據集的系統，源自 2016 年史丹佛大學的研究成果。在 Snorkel 中，可以在數小時或數天內開發大型訓練數據集，節省人工標註時間。
-   Snorkel 目前公開了三個關鍵的程序化操作：
    -   標記數據 **Labeling data** :
        
        -   例如使用啟發式規則或遠程監督技術。
        -   以撰寫 Python 函數的方式撰寫分類條件。  
            ![](https://i.imgur.com/Q1Of6KT.png)
    -   轉換數據 **Transforming data** :
        
        -   例如旋轉或拉伸圖像以執行數據增強。
        -   數據增強常見於電腦視覺影像資料集，透過圖片隨機旋轉、變形等方式增加訓練資料，而文字也可以透過同義詞進行數據增強。
        -   使用生成模型來估計不同標記函數的精度，然後重新加權並組合他們的標籤以產生一組概率訓練標籤，有效解決新數據清洗和集成問題；  
            ![](https://i.imgur.com/l6449iT.png)
    -   資料切片 **Slicing data** :
        
        -   這些標籤用於訓練判別模型，將數據分成不同的關鍵子集以進行監控或有針對性的改進。  
            ![](https://i.imgur.com/lhLwzhj.png)

實際步驟
----

我們將完成五個基本步驟，透過官方範例 [YouTube comments 資料集](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/)示範，具體程式執行筆者將官網範例調整為可以在 Colab 執行的範例，您可以透過 Colab 實作 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1mgroggxRG_yuLw2OaBD8OBzDuBQ57ZTo?usp=drive_fs) 。

-   簡單定義3個標籤並接續後續流程:
    
    ```
    ABSTAIN = -1
    NOT_SPAM = 0
    SPAM = 1
    ```
    

### 1\. 編寫標籤函數 (LFs)：

-   將用 LFs 以編程方式標記我們未標記的數據集，而不是手動標記任何訓練數據。 以下為官方導覽介紹的3種LF函數寫法:
    
-   關鍵字判別、正規表達式判別與用外部模組判別。
    
    ```python
    from snorkel.labeling import labeling_function
    
    # 關鍵字'my'篩選
    @labeling_function()
    def lf_keyword_my(x):
        """Many spam comments talk about 'my channel', 'my video', etc."""
        return SPAM if "my" in x.text.lower() else ABSTAIN
    
    ```
    
    ```python
    #正規表達式篩選
    import re
    
    @labeling_function()
    def lf_regex_check_out(x):
        """Spam comments say 'check out my video', 'check it out', etc."""
        return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN
    
    ```
    
    ```python
    #結合模組篩選
    from textblob import TextBlob
    
    @labeling_function()
    def lf_textblob_polarity(x):
        """
        We use a third-party sentiment classification model, TextBlob.
        We combine this with the heuristic that non-spam comments are often positive.
        """
        return NOT_SPAM if TextBlob(x.text).sentiment.polarity > 0.3 else ABSTAIN
    
    ```
    
-   更多類型的標記函數（包括文本以外的數據模式），請參閱其他[官方範例](https://snorkel.org/use-cases/)和[實際示例](https://snorkel.org/resources/)。
    

### 2\. 建模和組合 LF：

-   將前述設定好的`LabelModel` LF 組合為 list，將 LFs 應用於偽標註的訓練數據。
-   由於標註函數 LFs 的準確度和相關性未知，輸出標籤可能會重疊和衝突。 `snorkel.labeling.model.LabelModel` 可以自動估計它們的準確性和相關性，重新加權和組合它們的標籤，並生成我們最終的乾淨、集成的訓練標籤集：
    
    ```python
    from snorkel.labeling.model import LabelModel
    from snorkel.labeling import PandasLFApplier
    
    # set LFs
    lfs = [ lf_keyword_my, 
            lf_regex_check_out, 
            lf_short_comment, 
            lf_textblob_polarity
            ]
    
    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    
    ```
    
-   由於前述`LabelModel`可能很多數據為標註結果為放棄標示狀態的`ABSTAIN = -1`，為清理訓練資料集，將明顯標註`SPAM`、`NOT_SPAM`的訓練資料集保留進行後去處理。
    
    ```python
    df_train = df_train[df_train.label != ABSTAIN]
    
    ```
    

### 3\. 編寫數據增強的TF函數

-   接著透過建立一個TF函數來增強這個標記的訓練集。
    
-   以下`get_synonyms()`用`nltk.wordnet`獲取單詞的同義詞。
    
    ```python
    import random
    import nltk
    from nltk.corpus import wordnet as wn
    
    nltk.download("wordnet", quiet=True)
    
    def get_synonyms(word):
        """Get the synonyms of word from Wordnet."""
        lemmas = set().union(
            *[s.lemmas() for s in wn.synsets(word)]
            )
        return list(
            set(l.name().lower().replace("_", " ") for l in lemmas) - {word}
            )
    
    ```
    
-   使用 TF `snorkel.augmentation.transformation_function` 做為裝飾子，自訂 `tf_replace_word_with_synonym()` 函數將生成的同義詞加入訓練資料集。
    
    ```python
    from snorkel.augmentation import transformation_function
    
    @transformation_function()
    def tf_replace_word_with_synonym(x):
        """Try to replace a random word with a synonym."""
        words = x.text.lower().split()
        idx = random.choice(range(len(words)))
        synonyms = get_synonyms(words[idx])
        if len(synonyms) > 0:
            x.text = " ".join(words[:idx] + 
                              [synonyms[0]] + 
                              words[idx + 1 :]
                              )
            return x
    
    ```
    
-   將自訂 TF 函數加入訓練數據集。
    
    ```python
    from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
    
    tf_policy = ApplyOnePolicy(n_per_original=2, keep_original=True)
    tf_applier = PandasTFApplier([tf_replace_word_with_synonym], tf_policy)
    df_train_augmented = tf_applier.apply(df_train)
    
    ```
    
-   更多數據增強的調整可參閱 [Spam TFs tutorial](https://snorkel.org/use-cases/02-spam-data-augmentation-tutorial)。
    

### 4\. 建立切片函數 Slicing Function

-   Snorkel 的 Slicing Function 可用以監控特定切片，以及透過針對不同切片增加特徵以提高模型性能。
    
-   延續 Youtube 評論之中可能有惡意連結的想法，為此撰寫一個查找可疑縮網址的程式，這對找出惡意垃圾評論可能很關鍵。設定好 SF 可監控此切片的性能：
    
    ```python
    from snorkel.slicing import slicing_function
    
    
    @slicing_function()
    def short_link(x):
        """
        Return whether text matches common pattern 
        for shortened ".ly" links.
        """
        return int(bool(re.search(r"\w+\.ly", x.text)))
    
    ```
    

### 5\. 訓練分類器

-   Snorkel 的最終目標是創建一個標註完成的訓練資料集，然後將其插入任意機器學習框架（例如 TensorFlow、Keras、PyTorch、Scikit-Learn、Ludwig、XGBoost），以訓練強大的機器學習模型。
-   接續範例，將前述第3步完成的訓練資料集`df_train_augmented`，以 Scikit-Learn 的 n-gram 邏輯回歸模型進行推論，完成整個運用Snorkel 弱監督分類模型。
    
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    
    train_text = df_train_augmented.text.tolist()
    X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)
    
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X=X_train, y=df_train_augmented.label.values)
    
    ```
    

小結
--

-   Snorkel 透過程式邏輯標註程 Labeling data ，透過數據增強方式自動化轉換數據 Transforming data ，並且可以切片監控特定子資料集 Slicing data ，好處是可以輕易地融入機械學習系統工作流程，並且有自動標註的好處，標註水準還不錯。
-   雖然好用，但官方範例較複雜，希望能整理一份方便使用的指引，供您後續標註資料的參考。

參考
--

-   [https://www.snorkel.org/](https://www.snorkel.org/)
-   [https://arxiv.org/abs/1812.00417](https://arxiv.org/abs/1812.00417)
-   [https://github.com/snorkel-team/snorkel-tutorials](https://github.com/snorkel-team/snorkel-tutorials)
