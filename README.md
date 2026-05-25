# Inverse Design & Recovery for Optical Glasses

Хакатон: восстановление состава оптических стёкол по свойствам и генерация новых
бессвинцовых high-index составов.

- **Recovery** (SCHOTT → SciGlass): ищет ближайшие реальные стёкла в базе SciGlass по
  `n_d`, `ν_d`, `ρ`, `T_g` — без «придуманных» составов.
- **Inverse design** (NSGA-II): многокритериальный поиск с surrogate-моделями
  `composition → n_d / ν_d / ρ / T_g`.
- **GAN** (cWGAN-GP): генеративная модель составов под целевые `n_d`, `ν_d`.
- **glass_tool**: CLI + Streamlit для внешних пользователей (recover/generate за одну
  команду).

Подробный отчёт: [`ОТЧЕТ_РАБОТЫ.md`](ОТЧЕТ_РАБОТЫ.md) · Инструкция инструмента:
[`ИНСТРУКЦИЯ_ИНСТРУМЕНТ.md`](ИНСТРУКЦИЯ_ИНСТРУМЕНТ.md).

---

## Быстрый старт (для внешних пользователей)

### 1. Установка

```bash
git clone https://github.com/Lenpka/New_optical_glasses_hackathon.git
cd New_optical_glasses_hackathon
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements-ml.txt
```

### 2. Положите данные и модели

Большие файлы **не лежат в репозитории** (см. `.gitignore`). Скачайте их отдельно или
соберите сами:

| Файл | Где взять | Куда положить |
|------|-----------|----------------|
| `select_SciGK.csv.zip` (~40 MB) | пакет [GlassPy](https://github.com/drcassar/glasspy) или релиз проекта | `data/select_SciGK.csv.zip` |
| `merged_data.parquet` (~55 MB) | релизный архив | в корень проекта |
| `output/gan_design/checkpoints/generator.pt` | релизный архив или обучить | как есть |
| `output/inverse_design_2h/forward_models/*.joblib` | релизный архив или обучить | как есть |
| `schott-optical-glass-overview-excel-format-en 202501113.xlsx` | сайт SCHOTT | корень проекта |

Готовый ZIP с моделями: соберите командой `python make_release_zip.py` (внутри будут
данные + код, см. `dist/glass_tool_release.zip`).

### 3. Проверка

```bash
python glass_tool.py check
```

Все строчки `[OK]` — можно работать.

### 4. Использование

#### Восстановить состав по свойствам

```bash
python glass_tool.py recover --nd 1.85 --vd 25
python glass_tool.py recover --nd 1.85 --vd 25 --density 4.1 --tg 520 --out result.json
```

#### Восстановить состав по марке SCHOTT

```bash
python glass_tool.py list-glasses
python glass_tool.py recover --glass "N-SF11"
```

#### Сгенерировать кандидаты (GAN + surrogate)

```bash
python glass_tool.py generate --nd 1.90 --vd 22 -n 400 --top 15 --out candidates.csv
```

#### Меню в терминале

```bash
python glass_tool.py interactive
```

#### Веб-интерфейс (Streamlit)

```bash
pip install streamlit
streamlit run glass_tool_app.py
```

---

## Что показывают графики

| Файл | Что показывает |
|------|----------------|
| `output/presentation/figures/presentation_nd_vd_distance.png` | Главный слайд: SciGlass + design-кандидаты в плоскости `n_d × ν_d`, цвет — distance, ТОП-10 выделен |
| `output/presentation/figures/uncertainty_vs_nd.png` | Рост неопределённости с n_d — две панели (design proxy + recovery uncertainty) |
| `output/inverse_design_2h/figures/pareto_nd_vd.png` | Pareto-front NSGA-II, ТОП-20 в кольцах |
| `output/inverse_design_2h/figures/design_evaluation_summary.png` | Две панели: пространство свойств и риск экстраполяции |
| `output/schott_match/figures/pca_schott_sciglass.png` | PCA: где SCHOTT в облаке SciGlass по 4 свойствам |
| `output_v3/figures/jaccard_histogram.png` | Распределение Jaccard MODE_A vs MODE_B — почему recovery ill-posed |
| `output_v3/figures/distance_attribution.png` | Доли вкладов `n_d`, `ν_d`, `ρ`, `T_g` в расстояние |
| `output/property_distributions/property_distributions_overlay.png` | Распределения свойств для design / recovery / SCHOTT |
| `figures/comparison/*.png` | Сравнение ML-моделей: метрики и parity |

Пересборка презентационных рисунков: `python make_presentation_assets.py`.
Пересборка распределений свойств: `python plot_inverse_property_distributions.py`.

---

## Структура репозитория

```
glass_tool/             # CLI и Streamlit обёртки (core/cli)
glass_tool.py           # Точка входа CLI
glass_tool_app.py       # Streamlit интерфейс
make_release_zip.py     # Собрать ZIP для передачи моделей

match_schott_sciglass.py     # Recovery v1 (baseline)
match_schott_sciglass_v2.py  # v2: adaptive pool, uncertainty
match_schott_sciglass_v3.py  # v3: Jaccard, локальные PLS/XGB
inverse_glass_design.py      # Surrogate + NSGA-II
gan_glass_design.py          # cWGAN-GP

plots.py                            # Общая библиотека графиков
viz_presentation.py                 # Единый стиль презентационных рисунков
make_presentation_assets.py         # Сборка слайдов
plot_inverse_property_distributions.py  # Распределения n_d/ν_d/ρ/T_g
compare_models.py                   # Сравнение ML-моделей

output/                 # Артефакты прогонов (модели и крупные CSV в .gitignore)
output_v2/, output_v3/  # Recovery эксперименты
figures/                # Сравнение моделей
ОТЧЕТ_РАБОТЫ.md         # Полный отчёт
ИНСТРУКЦИЯ_ИНСТРУМЕНТ.md  # Инструкция для пользователя
requirements-ml.txt
```

---

## Если хочется переобучить с нуля

```bash
# Recovery (122 SCHOTT)
python match_schott_sciglass_v3.py

# Inverse design: surrogate + NSGA-II (долго)
python inverse_glass_design.py --pop-size 400 --generations 520

# GAN
python gan_glass_design.py --epochs 400
```

---

## Ключевые выводы (коротко)

- Прямая задача `состав → n_d` решается хорошо (R² ≈ 0.95 на elements).
- Обратная задача **плохо обусловлена**: median Jaccard MODE_A vs MODE_B ≈ 0.026, 99% стёкол SCHOTT — ill-posed.
- `n_d > 1.80` и `dist > 1.4 (p95)` — зона риска экстраполяции surrogate.
- Лучшие бессвинцовые кандидаты (NSGA-II + GAN, PbO=0, n≥1.80) — в файле
  `output/presentation/TOP10_candidates.csv`.

---

## Лицензия

См. [`LICENSE`](LICENSE).
