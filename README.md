### Магистратура ИИ. ФКН. НИУ ВШЭ. Годовой проект, 1 курс.
### Команда № 73, Тема № 64

## Personalized Anime Character Generation

### Описание:

Кто не хочет себе аниме-девочку (или мальчика) со своим лицом? Хотим обучить модель, которая бы по текстовому описанию/картинке человека выдавала бы аниме-версию.

### Куратор: 

- Неудачина Ева (tg: [@cocosinca](https://t.me/cocosinca))

### Команда:

- Гончаров Антон Дмитриевич (tg: [@GonHunch](https://t.me/GonHunch))
- Чайчук Михаил Викторович (tg: [@chaychuk_mikhail](https://t.me/chaychuk_mikhail))

### Сборка Docker:

```
1. git clone git@github.com:HSE-AI-Master-s-Team73-1st-Year-Project/Team73-Annual-Project.git
2. cd path/to/clone/Team73-Annual-Project
3. docker-compose up -d --build
4. open "http://localhost:8501" in your browser tab
5. docker-compose down (when finished working)
```

### Структура репозитория:

- Файл [checkpoints.md](./checkpoints.md) — описание пройденных этапов работы над годовым проектом.
- Файл [baseline.md](./baseline.md) — описание процесса построения бейзлайна (IP-Adapter).
- Файл [dataset.md](./dataset.md) — информация об используемых данных.
- Файл [EDA.md](./EDA.md) — общее описание проведенного EDA.
- Папка [/data_analysis](./data_analysis/):
    - [create_celeba_blip_captions.py](./data_analysis/create_celeba_blip_captions.py) и [create_caption.py](./data_analysis/create_caption.py) — скрипты для создания описаний к изображениям из датасета. Подробнее в [EDA.md](./EDA.md).
    - [EDA_images.ipynb](./data_analysis/EDA_images.ipynb) и [EDA_texts.ipynb](./data_analysis/EDA_text.ipynb) — ноутбуки с EDA.
- Папка [/src](./src/) — основной код для обучения ip-адаптера и подсчета метрик.
- [ip_adapter_demo.ipynb](./ip_adapter_demo.ipynb) и [ip_adapter_metrics.ipynb](ip_adapter_metrics.ipynb) — ноутбуки с демонстрацией работы ip-adaptera и подсчетом метрик.
- Папка  [/fastapi_app](./fastapi_app/) — код для FastAPI-приложения.
- Папка [/streamlit_app](./streamlit_app/) — код для streamlit-приложения.

<!-- ### FastAPI App

В [/fastapi_app](./fastapi_app/) содержится код fastapi-приложения для загрузки и инференса ip-адаптеров. Реализованные методы и их устройство приведены в файле  [/fastapi_app/openapi.yaml](./fastapi_app/openapi.yaml).

Демонстрация работы с API:

 ![fastapi-demo](./assets/gifs/fastapi-demo.gif) -->

### Streamlit App

В директории [/streamlit_app](./streamlit_app/) содержится код streamlit приложения, которое позволяет загружать обученные ip-адаптеры и проводить их инференс. Для своей работы оно использует API, реализованное в [/fastapi_app](./fastapi_app/).

Ниже приведена демонстрация работы этого приложения:

 ![streamlit-demo](./assets/gifs/streamlit-demo.gif)
