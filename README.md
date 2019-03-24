**Классификатор на основе метода k-ближайших соседей**

Зависимости:
* python 3.6
* Click==7.0
* numpy==1.16.1
* pandas==0.23.0
* scipy==1.2.1

Для обучения и сохранения модели можно использовать скрипт:

`python -m scripts.build_model`

При этом данные для обучения (файлы train.csv, test.csv, sample_submission.csv) должны быть предварительно помещены в папку data

Для выполнения предсказаний:

`python -m scripts.make_predictions`

