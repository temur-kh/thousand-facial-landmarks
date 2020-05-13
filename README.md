## Thousand Facial Landmarks

Kaggle соревнование: https://www.kaggle.com/c/made-thousand-facial-landmarks/overview

Запуск кода для обучения и генерации сабмита:

```python hack_train.py --name "resnet50-fin" --data "./data" --gpu```

Список сабмитов на соревновании. Лучшее решение - второе c начала списка (не обращайте внимание на названия):
![Посылки](https://user-images.githubusercontent.com/20341995/81843585-73c15e80-9556-11ea-97c0-56c178c245cc.png)

Решение базируется на коде baseline решения: https://github.com/BorisLestsov/MADE/tree/master/contest1/unsupervised-landmarks-thousand-landmarks-contest

Главные добавленные изменения:
- Модель - resnet50, предобученная на ImageNet, без фризов слоев
- Оптимизатор - Adam(lr=1e-3)
- Скедулер - ReduceLROnPlateau(patience=6, factor=0.3)
- Кол-во эпох обучения ~= 30
- Размер батча = 256 (при 512 не хватает памяти в GPU)
- Аугментация тренировочных данных (см. ниже)

Примененные аугментации:
- RandomPadAndSize(percent=0.15) - добавляет паддинг и ресайзит обратно в размер 128х128
- RandomRotate(ma_angle=15) - вращение
- ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.03)
- RandomGrayscale(p=0.1)

Также пробовал:
- RandomHorizontalFlip(p=0.5). Подробнее в ноутбуке `augmentations-review.ipynb`, но коротко это делало только хуже
- DropoutAugmentor(p=(0., 0.01)) - на подобие SaltAndPepper из библиотеки `imgaug`. Но у меня не хватило времени протестировать ее влияние, поэтому не стал рисковать и использовать для финального сабмишена.

Некоторые моменты, которые могут помочь улучшить данное решение и которые я все же не протестировал:
- использовать DropoutAugmentor. Думаю, это помогло бы не переобучаться на данных.
- заимлементить RandomAffine, в частности операцию Shear. Это могло бы помочь сгенерировать случаи, когда лицо сильно повернуто.
- поиграться с разными моделями, оптимизаторами и скедулерами. Увы, у меня не хватило времени и ресурсов перепробовать все.
