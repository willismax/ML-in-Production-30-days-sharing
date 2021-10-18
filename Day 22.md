# [Day 22 : 模型優化 - 知識蒸餾 Knowledge Distillation](https://ithelp.ithome.com.tw/articles/10268783)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10268783)

什麼是知識蒸餾 Knowledge Distillation
------------------------------

-   知識蒸餾 Knowledge Distillation 為模型壓縮技術，其中 student 模型從可以更複雜的 teacher 模型中 "學習" 。換言之，如果已經透過複雜的結構建構出不錯的模型，可以用知識蒸餾訓練出較簡易版本的模型，準確度不會差太多。
-   知識蒸餾主要運用在分類任務上。
-   Colab 支援 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1R1EQrUEP2Sb5gq-dIf_wbyA5KOhtRBWv)，參考[Keras官方範例](https://www.tensorflow.org/lite/performance/post_training_quantization)修改而成，理論請參見[論文](https://arxiv.org/abs/1503.02531)。

實作知識蒸餾 Knowledge Distillation
-----------------------------

-   本範例皆以 `tf.Kreas`實作，過程包含:
    1.  自定義一個`Distiller`類別。
    2.  用 CNN 訓練 teacher 模型。
    3.  student 模型向 teacher 學習。
    4.  訓練一個沒向老師學的 student_scratch 模型進行比較。

### 準備資料

-   延續前篇採用 [MNIST](https://keras.io/api/datasets/mnist/) ，您也可以改為 [CIFAR-10](https://keras.io/api/datasets/cifar10/)、 [cats vs dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) 等分類任務。

### 建立Distiller類別

-   本篇使用 Keras 官方範例定義的 `Distiller` 類別。
    
-   該類別繼承於 `th.keras.Model`，並改寫以下方法:
    
    -   `compile`：這個模型需要一些額外的參數來編譯，比如老師和學生的損失，alpha 和 temp 。
    -   `train_step`：控制模型的訓練方式。這將是真正的知識蒸餾邏輯所在。這個方法就是你做的時候調用的方法model.fit。
    -   `test_step`：控制模型的評估。這個方法就是你做的時候調用的方法model.evaluate。
    
    ```python
    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super(Distiller, self).__init__()
            self.teacher = teacher
            self.student = student
    
        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
            ):
            """ Configure the distiller.
            Args:
                optimizer: Keras optimizer for the student weights.
                metrics: Keras metrics for evaluation.
                student_loss_fn: Loss function of difference between student
                    predictions and ground-truth.
                distillation_loss_fn: Loss function of difference between soft
                    student predictions and soft teacher predictions.
                alpha: weight to student_loss_fn and 1-alpha to 
                    distillation_loss_fn.
                temperature: Temperature for softening probability 
                    distributions.
                    Larger temperature gives softer distributions.
            """
            super(Distiller, self).compile(
                optimizer=optimizer, 
                metrics=metrics
                )
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature
    
        def train_step(self, data):
            # Unpack data
            x, y = data
    
            # Forward pass of teacher
            teacher_predictions = self.teacher(x, training=False)
    
            with tf.GradientTape() as tape:
                # Forward pass of student
                student_predictions = self.student(x, training=True)
    
                # Compute losses
                student_loss = self.student_loss_fn(y, student_predictions)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(
                        teacher_predictions / self.temperature, axis=1
                        ),
                    tf.nn.softmax(
                        student_predictions / self.temperature, axis=1
                        )
                    )
                loss = self.alpha * student_loss + (
                    1 - self.alpha) * distillation_loss
    
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
    
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
            # Update the metrics configured in `compile()`.
            self.compiled_metrics.update_state(y, student_predictions)
    
            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"student_loss": student_loss, 
                 "distillation_loss": distillation_loss}
            )
            return results
    
        def test_step(self, data):
            # Unpack the data
            x, y = data
    
            # Compute predictions
            y_prediction = self.student(x, training=False)
    
            # Calculate the loss
            student_loss = self.student_loss_fn(y, y_prediction)
    
            # Update the metrics.
            self.compiled_metrics.update_state(y, y_prediction)
    
            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss})
            return results
    
    ```
    

### 建立老師與學生模型

-   提醒2件事情：
    
    -   最後一層沒有使用激勵函數 `softmax` ，因為知識蒸餾需要原始的權重分佈特徵，請記得去掉這層。
    -   通過 dropout 層的正則化將應用於教師而不是學生。這是因為學生應該能夠通過蒸餾過程學習這種正則化。
-   可以將學生模型視為教師模型的簡化（或壓縮）版本。
    
    ```python
    def big_model_builder():
      keras = tf.keras
      model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(
            filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
      ])
      return model
    
    def small_model_builder():
      keras = tf.keras
      model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(
            filters=12, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
      ])
      return model
    
    teacher = big_model_builder()
    student = small_model_builder()
    student_scratch = small_model_builder()
    
    ```
    

訓練老師
----

-   一如既往，毫無懸念的訓練原始模型/老師模型。
    
    ```python
    # Train teacher as usual
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    teacher.summary()
    
    # Train and evaluate teacher on data.
    teacher.fit(train_images, train_labels, epochs=2)
    _ , ACCURACY['teacher model'] = teacher.evaluate(test_images, test_labels)
    
    ```
    

### 透過知識蒸餾訓練學生

-   創建`Distiller`類別的實例並傳入學生和教師模型`distiller = Distiller(student=student, teacher=teacher)`。然後用合適的參數編譯並訓練。
    
    ```python
    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    
    # Distill teacher to student
    distiller.fit(
        train_images, 
        train_labels, 
        epochs=2, 
        shuffle=False
        )
    
    # Evaluate student on test dataset
    ACCURACY['distiller student model'], _ = distiller.evaluate(
        test_images, test_labels)
    
    
    ```
    

### 比較模型 \- 從頭訓練學生

-   student_scratch 是個學生自己訓練，未參與知識蒸餾過程的普通模型，架構與 student 相同，用來比較訓練成果。
    
    ```python
    # Train student as doen usually
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    student_scratch.summary()
    
    # Train and evaluate student trained from scratch.
    student_scratch.fit(
        train_images, 
        train_labels, 
        epochs=2, 
        shuffle=False
        )
    # student_scratch.evaluate(x_test, y_test)
    _, ACCURACY['student from scrath model'] = student_scratch.evaluate(
        test_images, 
        test_labels
        )
    
    ```
### 比較模型準確率
-   最終模型準確率約為:
    ```python
    ACCURACY
    {'teacher model': 0.9822999835014343,
     'distiller student model': 0.9729999899864197,
     'student from scrath model': 0.9697999954223633}
    ``` 
-   老師的準確率通常應該會高於學生，畢竟是傾注心力的模型。
-   「接受知識蒸餾的學生」表現通常會優於「自己從頭開始的學生」。
-   學生的模型雖然較簡易，知識蒸餾甚至會青出於藍勝於藍的情況，而且模型也較輕量。

小結
--

-   在遇到巨型模型(如: GTP-3)時，運算資源恐怕不容許您輕易部署上線，此時採用知識蒸餾，讓「學生」學習「老師」，至少比學生自主學習容易取得較佳結果。
-   也因為 Keras 官方範例模型用 Colab 跑較久，故也自己改寫較快收到成果的版本。
-   連續談自動化建模與模型優化，希望能讓您將模型上線更有信心，當然如何監控與觀察模型也相當重要，我們下篇見。  
    ![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif)

參考
--

-   [Keras knowledge_distillation](https://keras.io/examples/vision/knowledge_distillation/)
-   [知識蒸餾介紹](https://chtseng.wordpress.com/2020/05/12/%E7%9F%A5%E8%AD%98%E8%92%B8%E9%A4%BE-knowledgedistillation/)
