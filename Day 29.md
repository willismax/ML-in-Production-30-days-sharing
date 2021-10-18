# [Day 29 : 用於生產的 TensorFlow Extended (TFX) 實作](https://ithelp.ithome.com.tw/articles/10273652)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10273652)


-   用於生產的機械學習系統，在 [Day 28](https://ithelp.ithome.com.tw/articles/10272958) 介紹 TensorFlow Extended (TFX) 解決方案，是專門用於可擴充的高效能機器學習工作，包括建立模型、進行訓練、提供推論，以及管理線上、行動裝置 (TensorFlow Lite)和網頁應用服務 (TensorFlow JS) 的部署。
-   本日我們修改 TFX 官方在筆記本可執行的互動式範例，透過實作理解 TFX 如何在工作流程中透過組件功能驗證資料、特徵工程、訓練模型、模型分析到部署模型。
-   [Colab 實作範例 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1o4lRoAdpPkfCL6WV3X6JwXK5C27itbI6?usp=drive_fs)，內容源自[TFX 官方範例](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras)。

TFX 組件筆記本互動實作
-------------

### 0\. 安裝與設置 TFX 環境

-   更新 Colab 的 `pip`。
-   安裝 TFX，**安裝完需重新啟動執行階段(Restart Runtime)**
    
    ```python
    !pip install tfx
    
    ```
    
-   設定工作路徑。
-   下載[芝加哥 Taxi Trips 資料集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) ，將集構建一個預測小費`tips`的模型。

### 0\. 創建互動文件檢視器 InteractiveContext

-   `tfx.orchestration.experimental.interactive.interactive_context.InteractiveContext` 允許在 notebook 環境中以互動方式查看 TFX 組件。
-   `InteractiveContext` 預設使用臨時的中繼資料。
    -   已有自己的 pipeline 可設定 `pipe_root` 參數。
    -   已有中繼資料庫可設定 `metadata_connection_config` 參數。
-   透過 `InteractiveContext()` 逐一演示互動情形，請留意 `InteractiveContext.run()` 是在筆記本互動顯示方式，在生產環境中可專注組件流程（請參閱[構建 TFX 管道指南](https://www.tensorflow.org/tfx/guide/build_tfx_pipeline)）。

### 1\. ExampleGen

-   `ExampleGen` 將數據拆分為訓練集和評估集（預設為 2/3 訓練 、 1/3 評估）
-   `ExampleGen` 將數據轉換為 `tf.Example` 格式（參閱[說明](https://www.tensorflow.org/tutorials/load_data/tfrecord)）。
-   本範例將 `_data_root` 的 CSV 資料集輸入至 `ExampleGen`。
    
    ```python
    example_gen = tfx.components.CsvExampleGen(input_base=_data_root)
    context.run(example_gen)
    
    ```
    
-   觀察互動介面內容，建議您使用 Colab 測試，在此介面中:
    -   `.execution_id` 是持續累加的版次編號，在對應的資料夾將各版次的內容留存紀錄。
    -   `.component` 指的是該 TFX 組件，譬如下圖顯示為 `.component.CsvExampleGen`。
    -   `.component.inputs` 紀錄輸入來源。
    -   `.component.outputs` 紀錄輸出結果。  
        ![](https://i.imgur.com/KDJmdzf.png)

### 2\. StatisticsGen

-   `StatisticsGen` 組件輸入 `ExampleGen` 數據後，將據以計算出資料集的統計數據。
-   `StatisticsGen` 是 [TFDV](https://www.tensorflow.org/tfx/data_validation/get_started) 模組功能之一。
    
    ```python
    statistics_gen = tfx.components.StatisticsGen(
        examples = example_gen.outputs['examples']
        )
    context.run(statistics_gen)
    
    ```
    
-   `context.run(statistics_gen)` 觀察互動介面:
    -   `.execution_id` 版次累加至2。
    -   `.component.inputs` 組件輸入為 `Examples` 。
    -   輸出為 `ExampleStatistics` 。  
        ![](https://i.imgur.com/ylMd6h0.png)
-   `context.show(statistics_gen.outputs['statistics'])` 如同 TFDV 工具以 [Facets](https://pair-code.github.io/facets/) 視覺化統計資訊。
    
    ```python
    context.show(statistics_gen.outputs['statistics'])
    
    ```
    
    -   可以觀察判讀可能異常的紅色值、資料分佈情形等。  
        ![](https://i.imgur.com/YlUKeuk.png)

### 3\. SchemaGen

-   `SchemaGen`組件會依據您的資料統計自動產生 Schema ，包含數據預期邊界、資料類型與屬性它還使用[TensorFlow 數據驗證](https://www.tensorflow.org/tfx/data_validation/get_started)庫。
-   `SchemaGen` 同樣是 [TFDV](https://www.tensorflow.org/tfx/data_validation/get_started) 模組功能之一。
-   即便 Schema 自動生成已經很實用，但您仍應該會依據需求進行審查和修改。
    
    ```python
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)
    context.run(schema_gen)
    
    ```
    
    -   `SchemaGen` 輸入為 `StatisticsGen`，默認情況下查看已拆分的訓練資料集。
    -   `SchemaGen` 輸出為 `Schema` 。  
        ![](https://i.imgur.com/n8bCnxI.png)
    -   `SchemaGen` 執行後可透過 `context.show(schema_gen.outputs['schema'])` 查看 Schema 表格。
    -   表格呈現各特徵名稱、屬性、是否必須、所有值、Domain 及 邊界範圍等， 參見 [SchemaGen 文件](https://www.tensorflow.org/tfx/guide/schemagen)。  
        ![](https://i.imgur.com/8CkuVUg.png)  
        ![](https://i.imgur.com/9CF3JOx.png)

### 4\. ExampleValidator

-   `ExampleValidator` 組件根據 Schema 的預期檢測數據中的異常。
-   `ExampleValidator` 同樣是 [TFDV](https://www.tensorflow.org/tfx/data_validation/get_started) 模組功能之一。
-   `ExampleValidator` 的輸入是來自具有數據統計資訊的 `StatisticsGen` 以及具有數據定義 Schema 的 `SchemaGen`。
-   `ExampleValidator` 的輸出 `anomalies` 是有無異常的判讀結果。
    
    ```python
    example_validator = tfx.components.ExampleValidator(
        statistics = statistics_gen.outputs['statistics'],
        schema = schema_gen.outputs['schema'])
    context.run(example_validator)
    
    ```
    
    -   執行 `ExampleValidator` 後可以產生異常情形的圖表，綠字 No anomalies found. 表示無異常。  
        ![](https://i.imgur.com/1q8JWQR.png)
        
        ```python
        context.show(example_validator.outputs['anomalies'])
        
        ```
        
        ![](https://i.imgur.com/yiwJs29.png)

### 5\. Transform

-   `Transform` 組件為訓練和服務執行特徵工程。
    
-   `Transform` 使用[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) 模組。
    
-   `Transform` 輸入數據來自 `ExampleGen` 、 Schema 來自 `SchemaGen` ，以及自行定義如何進行特徵工程的模組。
    
-   以下為自行定義的 Transform 程式碼範例，（有關 TensorFlow Transform API 的介紹，[請參閱教程](https://www.tensorflow.org/tfx/tutorials/transform/simple)）。
    
-   Notebook 魔術指令 `%%writefile` ，可以將 cell 內的程式碼指定保存為檔案，該檔案可以用 `Transform` 組件將程式碼檔案做為模組輸入執行。
    
    ```python
    %%writefile {_taxi_constants_module_file}
    
    # 假設分類特徵的最大值
    MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
    
    CATEGORICAL_FEATURE_KEYS = [
        'trip_start_hour', 
        'trip_start_day', 
        'trip_start_month',
        'pickup_census_tract', 
        'dropoff_census_tract', 
        'pickup_community_area',
        'dropoff_community_area'
        ]
    
    DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']
    
    # tf.transform用於編碼每個特徵的桶數=10
    FEATURE_BUCKET_COUNT = 10
    
    BUCKET_FEATURE_KEYS = [
        'pickup_latitude', 
        'pickup_longitude', 
        'dropoff_latitude',
        'dropoff_longitude'
        ]
    
    # tf.transform用於編碼VOCAB_FEATURES的詞彙術語數量=1000
    VOCAB_SIZE = 1000
    
    # Count of out-of-vocab buckets in which unrecognized 
    # VOCAB_FEATURES are hashed.
    OOV_SIZE = 10
    
    VOCAB_FEATURE_KEYS = [
        'payment_type',
        'company',
    ]
    
    # Keys
    LABEL_KEY = 'tips'
    FARE_KEY = 'fare'
    
    def transformed_name(key):
      return key + '_xf'
    
    ```
    
-   接著編寫 `preprocessing_fn` 將原始數據轉換特徵。
    
    ```python
    %%writefile {_taxi_transform_module_file}
    
    import tensorflow as tf
    import tensorflow_transform as tft
    
    import taxi_constants
    
    _DENSE_FLOAT_FEATURE_KEYS = taxi_constants.DENSE_FLOAT_FEATURE_KEYS
    _VOCAB_FEATURE_KEYS = taxi_constants.VOCAB_FEATURE_KEYS
    _VOCAB_SIZE = taxi_constants.VOCAB_SIZE
    _OOV_SIZE = taxi_constants.OOV_SIZE
    _FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT
    _BUCKET_FEATURE_KEYS = taxi_constants.BUCKET_FEATURE_KEYS
    _CATEGORICAL_FEATURE_KEYS = taxi_constants.CATEGORICAL_FEATURE_KEYS
    _FARE_KEY = taxi_constants.FARE_KEY
    _LABEL_KEY = taxi_constants.LABEL_KEY
    _transformed_name = taxi_constants.transformed_name
    
    
    def preprocessing_fn(inputs):
      """tf.transform's callback function for preprocessing inputs.
      Args:
        inputs: map from feature keys to raw not-yet-transformed 
        features.
      Returns:
        Map from string feature key to transformed feature 
        operations.
      """
      outputs = {}
      for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, 
        # setting nan's to the mean.
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))
    
      for key in _VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)
    
      for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)
    
      for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])
    
      # Was this passenger a big tipper?
      taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
      tips = _fill_in_missing(inputs[_LABEL_KEY])
      outputs[_transformed_name(_LABEL_KEY)] = tf.where(
          tf.math.is_nan(taxi_fare),
          tf.cast(
              tf.zeros_like(taxi_fare), 
              tf.int64
              ),
          # Test if the tip was > 20% of the fare.
          tf.cast(
              tf.greater(
              tips, 
              tf.multiply(taxi_fare, tf.constant(0.2))), 
              tf.int64
              )
          )
    
      return outputs
    
    
    def _fill_in_missing(x):
      """Replace missing values in a SparseTensor.
      Fills in missing values of `x` with '' or 0, 
      and converts to a dense tensor.
    
      Args:
        x: A `SparseTensor` of rank 2.  
        Its dense shape should have size at most 1 in the second dimension.
      Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
      """
      if not isinstance(x, tf.sparse.SparseTensor):
        return x
    
      default_value = '' if x.dtype == tf.string else 0
      return tf.squeeze(
          tf.sparse.to_dense(
              tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
              default_value),
              axis=1
              )
    
    ```
    
-   將特徵工程程式傳遞給 `Transform` 組件轉換資料。
    
    ```python
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(_taxi_transform_module_file))
    context.run(transform)
    
    ```
    
-   `Transform` 組件將產生以下兩種類型的輸出：
    
    -   `transform_graph` 是可以執行預處理操作的圖（此圖將包含在服務和評估模型中）。
    -   `transformed_examples` 表示預處理的訓練和評估數據。  
        ![](https://i.imgur.com/OvLbozG.png)
    -   查看`transform.outputs` 。  
        ![](https://i.imgur.com/QbbspS8.png)
-   輸出的 `transform_graph` 同時指向包含3個子目錄的目錄。
    
    -   `transformed_metadata`子目錄包含預處理數據的架構。
    -   `transform_fn`子目錄包含實際的預處理圖。
    -   `metadata`子目錄包含原始數據的架構。
    
    ```python
    train_uri = transform.outputs['transform_graph'].get()[0].uri
    os.listdir(train_uri)
    
    ```
    

### 6\. Trainer

-   `Trainer`組件負責訓練 TensorFlow 模型。
    
-   `Trainer` 預設使用 Estimator API ，如要使用 Keras API，您需要通過在 Trainer 的構造函數中設置來指定 `custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor)` ，參閱[Generic Trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md) 。
    
-   `Trainer` 的輸入來源:
    
    -   來自 `SchemaGen` 的 Schema。
    -   來自 `Transform` 的 graph。
    -   訓練參數。
    -   做為模組輸入的自定義程式碼。
-   以下為用戶自定義模型代碼示範（[參見 TensorFlow Keras API 介紹](https://www.tensorflow.org/guide/keras)）。
    
    ```python
    %%writefile {_taxi_trainer_module_file}
    
    from typing import List, Text
    
    import os
    import absl
    import datetime
    import tensorflow as tf
    import tensorflow_transform as tft
    
    from tfx import v1 as tfx
    from tfx_bsl.public import tfxio
    
    import taxi_constants
    
    _DENSE_FLOAT_FEATURE_KEYS = taxi_constants.DENSE_FLOAT_FEATURE_KEYS
    _VOCAB_FEATURE_KEYS = taxi_constants.VOCAB_FEATURE_KEYS
    _VOCAB_SIZE = taxi_constants.VOCAB_SIZE
    _OOV_SIZE = taxi_constants.OOV_SIZE
    _FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT
    _BUCKET_FEATURE_KEYS = taxi_constants.BUCKET_FEATURE_KEYS
    _CATEGORICAL_FEATURE_KEYS = taxi_constants.CATEGORICAL_FEATURE_KEYS
    _MAX_CATEGORICAL_FEATURE_VALUES = taxi_constants.MAX_CATEGORICAL_FEATURE_VALUES
    _LABEL_KEY = taxi_constants.LABEL_KEY
    _transformed_name = taxi_constants.transformed_name
    
    
    def _transformed_names(keys):
      return [_transformed_name(key) for key in keys]
    
    
    def _get_serve_tf_examples_fn(model, tf_transform_output):
      """Returns a function that parses a serialized tf.Example and applies TFT."""
    
      model.tft_layer = tf_transform_output.transform_features_layer()
    
      @tf.function
      def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    
      return serve_tf_examples_fn
    
    
    def _input_fn(
        file_pattern: List[Text],
        data_accessor: tfx.components.DataAccessor,
        tf_transform_output: tft.TFTransformOutput,
        batch_size: int = 200) -> tf.data.Dataset:
      """Generates features and label for tuning/training.
    
      Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch
    
      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
      """
      return data_accessor.tf_dataset_factory(
          file_pattern,
          tfxio.TensorFlowDatasetOptions(
              batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
          tf_transform_output.transformed_metadata.schema)
    
    
    def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
      """Creates a DNN Keras model for classifying taxi data.
      Args:
        hidden_units: [int], the layer sizes of the DNN (input layer first).
      Returns:
        A keras Model.
      """
      real_valued_columns = [
          tf.feature_column.numeric_column(key, shape=())
          for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
      ]
      categorical_columns = [
          tf.feature_column.categorical_column_with_identity(
              key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
          for key in _transformed_names(_VOCAB_FEATURE_KEYS)
      ]
      categorical_columns += [
          tf.feature_column.categorical_column_with_identity(
              key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
          for key in _transformed_names(_BUCKET_FEATURE_KEYS)
      ]
      categorical_columns += [
          tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
              key,
              num_buckets=num_buckets,
              default_value=0) for key, num_buckets in zip(
                  _transformed_names(_CATEGORICAL_FEATURE_KEYS),
                  _MAX_CATEGORICAL_FEATURE_VALUES)
      ]
      indicator_column = [
          tf.feature_column.indicator_column(categorical_column)
          for categorical_column in categorical_columns
      ]
    
      model = _wide_and_deep_classifier(
          # TODO(b/139668410) replace with premade wide_and_deep keras model
          wide_columns=indicator_column,
          deep_columns=real_valued_columns,
          dnn_hidden_units=hidden_units or [100, 70, 50, 25])
      return model
    
    
    def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
      """Build a simple keras wide and deep model.
    
      Args:
        wide_columns: Feature columns wrapped in indicator_column for wide (linear)
          part of the model.
        deep_columns: Feature columns for deep part of the model.
        dnn_hidden_units: [int], the layer sizes of the hidden DNN.
    
      Returns:
        A Wide and Deep Keras model
      """
      # Following values are hard coded for simplicity in this example,
      # However prefarably they should be passsed in as hparams.
    
      # Keras needs the feature definitions at compile time.
      # TODO(b/139081439): Automate generation of input layers from FeatureColumn.
      input_layers = {
          colname: tf.keras.layers.Input(
              name=colname, 
              shape=(), 
              dtype=tf.float32
              )
          for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
      }
      input_layers.update({
          colname: tf.keras.layers.Input(
              name=colname, 
              shape=(), 
              dtype='int32')
          for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
      })
      input_layers.update({
          colname: tf.keras.layers.Input(
              name=colname, 
              shape=(), 
              dtype='int32')
          for colname in _transformed_names(_BUCKET_FEATURE_KEYS)
      })
      input_layers.update({
          colname: tf.keras.layers.Input(
              name=colname, 
              shape=(), 
              dtype='int32')
          for colname in _transformed_names(_CATEGORICAL_FEATURE_KEYS)
      })
    
      # TODO(b/161952382): Replace with Keras preprocessing layers.
      deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
      for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
      wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
    
      output = tf.keras.layers.Dense(1)(
              tf.keras.layers.concatenate([deep, wide]))
    
      model = tf.keras.Model(input_layers, output)
      model.compile(
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.Adam(lr=0.001),
          metrics=[tf.keras.metrics.BinaryAccuracy()])
      model.summary(print_fn=absl.logging.info)
      return model
    
    
    # TFX Trainer will call this function.
    def run_fn(fn_args: tfx.components.FnArgs):
      """Train the model based on given args.
      Args:
        fn_args: Holds args used to train the model as name/value pairs.
      """
      # Number of nodes in the first layer of the DNN
      first_dnn_layer_size = 100
      num_dnn_layers = 4
      dnn_decay_factor = 0.7
    
      tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
      train_dataset = _input_fn(
          fn_args.train_files, 
          fn_args.data_accessor, 
          tf_transform_output, 
          40
          )
      eval_dataset = _input_fn(
          fn_args.eval_files, 
          fn_args.data_accessor, 
          tf_transform_output, 
          40
          )
    
      model = _build_keras_model(
          # Construct layers sizes with exponetial decay
          hidden_units=[
              max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
              for i in range(num_dnn_layers)
          ])
    
      tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=fn_args.model_run_dir, update_freq='batch')
      model.fit(
          train_dataset,
          steps_per_epoch=fn_args.train_steps,
          validation_data=eval_dataset,
          validation_steps=fn_args.eval_steps,
          callbacks=[tensorboard_callback])
    
      signatures = {
          'serving_default':
              _get_serve_tf_examples_fn(
                  model,
                  tf_transform_output).get_concrete_function(
                      tf.TensorSpec(
                          shape=[None],
                          dtype=tf.string,
                          name='examples'
                          )
                      ),
                    }
      model.save(
          fn_args.serving_model_dir, 
          save_format='tf', 
          signatures=signatures
          )
    
    ```
    
-   創立 `taxi_trainer.py` 之後將程式碼做為模組傳遞給 `Trainer` 組件並運行它來訓練模型。
    
    ```python
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(_taxi_trainer_module_file),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=10000),
        eval_args=tfx.proto.EvalArgs(num_steps=5000))
    context.run(trainer)
    
    ```
    
    -   互動內文如下:  
        ![](https://i.imgur.com/9J6kfJn.png)

### (選用) 以 TensorBoard 分析訓練模型

-   可以透過 TensorBoard 分析模型訓練曲線。
    
    ```python
    model_run_artifact_dir = trainer.outputs['model_run'].get()[0].uri
    
    %load_ext tensorboard
    %tensorboard --logdir {model_run_artifact_dir}
    
    ```
    
    -   視覺化結果如下:  
        ![](https://i.imgur.com/v1wd1rq.png)

### 7\. Evaluator

-   `Evaluator` 組件可評估模型性能。
-   `Evaluator` 組件為 [TensorFlow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/model_analysis/get_started) 模組功能。
-   `Evaluator` 可以設定門檻值以比較並選擇較佳的模型。這在生產管道設置中很有用，您可以每天自動訓練和驗證模型。
    
    ```python
    model_resolver = tfx.dsl.Resolver(
          strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
          model=tfx.dsl.Channel(
              type=tfx.types.standard_artifacts.Model),
          model_blessing=tfx.dsl.Channel(
              type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                  'latest_blessed_model_resolver')
    context.run(model_resolver)
    
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)
    context.run(evaluator)
    
    ```
    
-   `Evaluator` 的輸入:
    -   輸入資料集來自 `ExampleGen`。
    -   訓練模型來自 `Trainer` 和切片配置。切片配置允許您根據特徵值對指標進行切片（例如，您的模型在早上 8 點和晚上 8 點開始的出租車行程中表現如何？）。  
        ![](https://i.imgur.com/t5OvYf4.png)
-   在此筆記本範例只訓練一個模型，所以`Evaluator`自動將模型標記為“Good”。
    
    ```python
    context.show(evaluator.outputs['evaluation'])
    
    ```
    
-   要切片顯示模型情形，需使用 TFMA 模組。
-   在此示範將`trip_start_hour`切片視覺化，TFMA 支援許多其他可視化，例如公平指標和繪製模型性能的時間序列。要了解更多信息，請參閱[教學](https://www.tensorflow.org/tfx/tutorials/model_analysis/tfma_basic)。
    
    ```python
    import tensorflow_model_analysis as tfma
    
    # Get the TFMA output result path and load the result.
    PATH_TO_RESULT = evaluator.outputs['evaluation'].get()[0].uri
    tfma_result = tfma.load_eval_result(PATH_TO_RESULT)
    
    # Show data sliced along feature column trip_start_hour.
    tfma.view.render_slicing_metrics(
        tfma_result, slicing_column='trip_start_hour')
    
    ```
    
    -   切片示範如下:  
        ![](https://i.imgur.com/FkBXcE1.png)
-   通過門檻值的模型會得到祝福 `blessing` ，第一次預設會自動取得，之後持續訓練過程會將取得祝福的模型再上線。
    
    ```python
    blessing_uri = evaluator.outputs['blessing'].get()[0].uri
    !ls -l {blessing_uri}
    
    ```
    
    ![](https://i.imgur.com/ipnkuaN.png)

### 8\. Pusher

-   `Pusher` 組件通常位於 TFX 管道末端。
    
-   `Pusher` 組件檢查模型是否已通過驗證，如果是，則將模型導出至 `_serving_model_dir`。
    
-   `Pusher` 將以 `SavedModel` 格式導出您的模型。
    
    ```python
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=_serving_model_dir)))
    context.run(pusher)
    
    ```
    
    ![](https://i.imgur.com/F3DC561.png)
    
-   終於完成 TFX 所有組件的示範!
    
-   如果在 GCP 在 AI Platform 以 Kubeflow 部署 TFX，Pipeline 進一步實作可參閱 [TFX on Google Cloud AI Platform Pipelines](https://www.qwiklabs.com/focuses/18244?locale=zh_TW&parent=catalog) 視覺化如下方便查閱。  
    ![](https://i.imgur.com/6P4RCS5.png)
    

![/images/emoticon/emoticon34.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon34.gif)

小結
--

-   在筆記本逐步互動式的完成 TFX 各組件的作業流程，實際上工作可以不用那麼複雜，在 GCP 的解決方案，可以用 Google Cloud AI Platform Pipeline ，是 Google Cloud AI Platform + TFX 容器化部署的結合，您可以參閱[TFX: Production ML with TensorFlow in 2020 (TF Dev Summit '20)](https://www.youtube.com/watch?v=I3MjuFGmJrg) 影片說明，理解 TFX 各組件串起的用於生產的機械學習工作流程。

參考
--

-   [TFX Keras Component Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras)。
-   [TFX on Google Cloud AI Platform Pipelines](https://www.qwiklabs.com/focuses/18244?locale=zh_TW&parent=catalog)
-   [TFX: Production ML with TensorFlow in 2020 (TF Dev Summit '20)](https://www.youtube.com/watch?v=I3MjuFGmJrg)
