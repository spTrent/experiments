# Evaluate на арендованной A10

В архиве лежат готовые ноутбуки частей 2, 3 и 4, исходные parquet, четыре OOF-output, полная сборка с winner и результат оценки части 1. Профиль five_starts, CODE_ID и параметры моделей совпадают с завершённой частью 1. Сервер создаёт прогнозы test для недостающих шести горизонтов. Три части запускаются последовательно на одной A10. Для этого этапа внешняя tuned Ridge и PyTorch не требуются.

## 1. Перенос с Mac

В терминале Mac подставьте свой SSH-логин, IP и порт из панели аренды. Если провайдер выдал ключ, добавьте к scp и ssh параметр -i с путём к ключу. Ключ остаётся на Mac.

```bash
SERVER_USER_HOST='логин@адрес'
SERVER_PORT=22
scp -P "$SERVER_PORT" /Users/sptrent/projects/rwb_case_2/stacking/bundles/rwb-evaluate-server.zip "$SERVER_USER_HOST:~/"
ssh -p "$SERVER_PORT" "$SERVER_USER_HOST"
```

Далее команды выполняются на сервере. Распакуйте архив на постоянный диск. Для стандартного варианта:

```bash
mkdir -p ~/rwb-evaluate
unzip -oq ~/rwb-evaluate-server.zip -d ~/rwb-evaluate
cd ~/rwb-evaluate
nvidia-smi
df -h .
```

Если постоянный диск смонтирован, например, в /workspace, распакуйте туда и выполните `export RWB_ROOT=/workspace/rwb-evaluate`, затем `cd "$RWB_ROOT"`. Если доступен только веб-интерфейс JupyterLab, можно загрузить тот же ZIP кнопкой Upload и распаковать его из терминала JupyterLab. Рабочая папка должна переживать перезапуск виртуальной машины.

```text
input/
  case_rwb/         исходные train и test, по одному файлу
  stack_oof/        part1 … part4 с оригинальными stack_<RUN_ID>
  stack-assemble/   сборка с meta_tuned/winner.json
  evaluate-part1/   готовые test для h1 и h8
output/             новые результаты и чекпойнты сервера
scratch/            временные матрицы признаков
```

## 2. Окружение и проверка

Если провайдер уже установил RAPIDS, CuPy, GPU CatBoost/XGBoost, LightGBM и Python 3.11+, активируйте это окружение и попробуйте `python check_server.py`. При успехе переходите к шагу 3. Скрипт проверяет хеши исходных parquet, совместимость победителя, наличие всех OOF и результатов части 1, а затем выполняет маленькие синтетические fit на GPU.

Для нового окружения приложен environment.yml для Linux x86_64, Python 3.12 и CUDA 12.9. Проверьте, что драйвер сервера поддерживает CUDA 12.9 или новее. Если conda уже есть, используйте её. Для установки Miniforge в домашнюю папку:

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge-rwb.sh
bash /tmp/miniforge-rwb.sh -b -p "$HOME/miniforge-rwb"
source "$HOME/miniforge-rwb/etc/profile.d/conda.sh"
```

```bash
conda env create -f environment.yml
conda activate rwb-a10
python check_server.py
```

При другой конфигурации драйвера подберите совместимую версию RAPIDS/CUDA. [Официальная инструкция NVIDIA](https://docs.rapids.ai/install/). Для проверки только структуры входов доступна команда `python check_server.py --files-only`.

## 3. Запуск с сохранением прогресса

В активированном GPU-окружении и из каталога проекта:

```bash
nohup python -u run_evaluate.py 2 3 4 > evaluate.log 2>&1 < /dev/null &
tail -f evaluate.log
```

Ctrl+C завершает просмотр tail. Фоновый запуск продолжится после закрытия SSH. Для запуска только части 2 укажите `python -u run_evaluate.py 2`. Скрипт запускает каждую часть в отдельном процессе и переходит к следующей только после успешного завершения. После сбоя с тем же input/output повторите команду для недостающих частей. Готовые fit подхватываются из output, временные матрицы строятся заново. Не запускайте одновременно второй экземпляр для той же части.

Вариант через JupyterLab: зарегистрируйте окружение командой `python -m ipykernel install --user --name rwb-a10 --display-name "RWB A10"`, выберите этот kernel и последовательно выполните Run All в evaluate_part_2.ipynb, затем 3 и 4. Используйте один способ запуска.

## 4. Проверка и скачивание результатов

```bash
cat output/session_part*_five_starts_evaluate_worker.json
ls output/stack_0158a242d49e0c00c439657a/h*/test
```

Нужны три статуса complete и горизонты h2/h7, h3/h6, h4/h5. Ранее рассчитанные h1/h8 остаются в input/evaluate-part1. Для передачи полных test-результатов соберите их в один архив на сервере:

```bash
tar -czf evaluate-workers-all.tar.gz input/evaluate-part1 output
```

Скачайте с Mac, используя ту же пару адреса и порта:

```bash
scp -P "$SERVER_PORT" "$SERVER_USER_HOST:~/rwb-evaluate/evaluate-workers-all.tar.gz" .
```

При размещении в /workspace укажите соответствующий удалённый путь. После worker следующий этап — evaluate / collect части 1 с исходными данными, полным stack-assemble и всеми четырьмя test-результатами. Независимая оценка test должна быть сохранена до полного переобучения для сабмита.
