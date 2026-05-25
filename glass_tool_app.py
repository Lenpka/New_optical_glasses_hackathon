"""Веб-интерфейс для glass_tool (Streamlit).

    pip install streamlit
    streamlit run glass_tool_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError as e:
    raise SystemExit(
        "Установите Streamlit: pip install streamlit\n"
        "Запуск: streamlit run glass_tool_app.py"
    ) from e

from glass_tool.core import (
    ToolPaths,
    check_environment,
    format_generate_text,
    format_recover_text,
    generate_compositions,
    list_schott_glasses,
    recover_composition,
)

st.set_page_config(
    page_title="Составы оптических стекол",
    page_icon="🔬",
    layout="wide",
)

st.title("Инструмент составов оптических стекол")
st.caption(
    "Восстановление — реальные соседи SciGlass (без синтеза). "
    "Генерация — cWGAN-GP + проверка surrogate-моделями."
)

with st.sidebar:
    st.header("Пути к данным")
    paths = ToolPaths(
        sciglass_zip=Path(
            st.text_input("SciGlass zip", value=str(ToolPaths().sciglass_zip))
        ),
        schott_xlsx=Path(
            st.text_input("SCHOTT xlsx", value=str(ToolPaths().schott_xlsx))
        ),
        gan_dir=Path(st.text_input("GAN output", value=str(ToolPaths().gan_dir))),
        forward_models=Path(
            st.text_input("Forward models", value=str(ToolPaths().forward_models))
        ),
        merged_data=Path(st.text_input("merged_data.parquet", value=str(ToolPaths().merged_data))),
    )
    if st.button("Проверить окружение"):
        c = check_environment(paths)
        st.json(c)

tab_recover, tab_generate, tab_help = st.tabs(
    ["Восстановить состав", "Сгенерировать (GAN)", "Справка"]
)

with tab_recover:
    mode = st.radio("Режим", ["По свойствам", "По марке SCHOTT"], horizontal=True)

    if mode == "По свойствам":
        c1, c2, c3, c4 = st.columns(4)
        nd = c1.number_input("n_d", value=1.85, min_value=1.4, max_value=2.5, step=0.01)
        vd = c2.number_input("ν_d", value=25.0, min_value=10.0, max_value=80.0, step=0.5)
        use_rho = c3.checkbox("Указать ρ", value=False)
        density = c3.number_input("ρ, кг/м³", value=4.0, disabled=not use_rho) if use_rho else None
        use_tg = c4.checkbox("Указать T_g", value=False)
        tg = c4.number_input("T_g, °C", value=500.0, disabled=not use_tg) if use_tg else None
        glass_name = None
    else:
        nd = vd = density = tg = None
        examples = list_schott_glasses(paths, 30)
        glass_name = st.selectbox(
            "Марка SCHOTT",
            options=examples if examples else ["N-SF11"],
            index=0,
        )
        st.text_input("Или введите своё имя", key="custom_glass")

    k_neighbors = st.slider("Число соседей (top-k)", 5, 30, 20)

    if st.button("Восстановить", type="primary"):
        custom = st.session_state.get("custom_glass", "").strip()
        gname = custom or glass_name
        try:
            with st.spinner("Поиск в SciGlass…"):
                if mode == "По марке SCHOTT":
                    result = recover_composition(
                        glass_name=gname, k_neighbors=k_neighbors, paths=paths
                    )
                else:
                    result = recover_composition(
                        nd=nd,
                        vd=vd,
                        density=density if use_rho else None,
                        tg=tg if use_tg else None,
                        k_neighbors=k_neighbors,
                        paths=paths,
                    )
            st.success("Готово")
            st.markdown(f"**Основной состав:** `{result['primary_composition']}`")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("distance", f"{result['distance_first']:.3f}")
            m2.metric("Jaccard", f"{result['jaccard_topk']:.3f}")
            m3.metric("uncertainty", f"{result['uncertainty_score']:.3f}")
            m4.metric("физичен", "да" if result["best_neighbor_plausible"] else "нет")

            st.dataframe(pd.DataFrame(result["neighbors"]), use_container_width=True)
            with st.expander("Полный отчёт (текст)"):
                st.code(format_recover_text(result))
            st.download_button(
                "Скачать JSON",
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                file_name="recover_result.json",
            )
        except Exception as e:
            st.error(str(e))

with tab_generate:
    st.info(
        "Нужны обученные модели в `output/gan_design/` и `output/inverse_design_2h/forward_models/`."
    )
    g1, g2, g3 = st.columns(3)
    g_nd = g1.number_input("Целевой n_d ", value=1.90, min_value=1.7, max_value=2.2, step=0.01)
    g_vd = g2.number_input("Целевой ν_d ", value=22.0, min_value=12.0, max_value=50.0, step=0.5)
    g_n = g3.number_input("Генераций", value=300, min_value=50, max_value=2000, step=50)
    g_top = st.slider("Показать лучших", 5, 50, 15)

    if st.button("Сгенерировать", type="primary"):
        try:
            with st.spinner("cWGAN-GP + scoring…"):
                result = generate_compositions(
                    g_nd,
                    g_vd,
                    n_samples=int(g_n),
                    top_k=int(g_top),
                    paths=paths,
                )
            st.success(
                f"Feasible (всего): {result['n_feasible_all']} / {result['n_generated']}"
            )
            df = pd.DataFrame(result["candidates"])
            st.dataframe(df, use_container_width=True)
            st.caption(result["disclaimer"])
            st.download_button(
                "Скачать CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name="gan_candidates.csv",
            )
        except Exception as e:
            st.error(str(e))

with tab_help:
    st.markdown(
        """
### Восстановление
По заданным **n_d**, **ν_d** (и при необходимости ρ, T_g) ищутся ближайшие **реальные** стёкла SciGlass.
Ответ — состав **лучшего соседа**, не усреднение и не «придуманный» рецепт.

- **Jaccard** < 0.3 — задача плохо обусловлена (разные соседи при 2 vs 4 свойствах).
- **uncertainty** — разброс соседних составов × среднее расстояние.

### Генерация
cWGAN-GP создаёт новые составы под цель (n_d, ν_d); surrogate оценивает свойства и **distance** до обучающей выборки.

### CLI (без браузера)
```bash
python glass_tool.py check
python glass_tool.py recover --nd 1.85 --vd 25
python glass_tool.py recover --glass "N-SF11"
python glass_tool.py generate --nd 1.90 --vd 22 -n 400 --out out.csv
python glass_tool.py interactive
```
        """
    )
