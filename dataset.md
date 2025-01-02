Ссылка на данные: [https://disk.yandex.ru/d/Yg91Vb-4jW2x_A](https://disk.yandex.ru/d/QQFxCLQynOE_pg)

В данном проекте мы используем датасет [Multi-Modal-CelebA-HQ](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset).

Данный датасет содержит 30000 изображений лиц различных знаменитостей в разрешении $1024 \times 1024$.

Каждому изображению соответсует 10 текстовых описаний, составленных авторами оригинального датасета на основе меток атрибутов картинок. Эти описания содержат информацию о различных характеристиках внешности человека на фото.
Дополнительно для каждой картинки мы самостоятельно сгенерировали текстовое описание при помощи модели [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) с Hugging Face.

Внутри архива с датасетом содержатся:
- Папка CelebA-HQ-img — сами изображения
- Файл captions.csv — описания для картинок в формате таблицы
- Файл CelebAMask-HQ-attribute-anno.txt — метки атрибутов для каждой из картинок

Скрипт для генерации описаний при помощи Blip2 находится в файле [data_analysis/сreate_celeba_blip_captions.py](./data_analysis/create_celeba_blip_captions.py)
Скрипт для генерации 10 текстов по атрибутам от авторов оригинального датасета находится в файле [data_analysis/create_caption.py](./data_analysis/create_caption.py)
