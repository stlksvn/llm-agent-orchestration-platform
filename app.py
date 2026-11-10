
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_orchestrator.models import ChatMessage, LLMResponse  # noqa: E402
from agent_orchestrator.providers import OllamaProvider  # noqa: E402
from agent_orchestrator.benchmark.runner import (  # noqa: E402
    STRATEGIES as BENCHMARK_STRATEGIES,
    load_tasks,
    run_benchmark,
    summarize,
    write_results,
)
from agent_orchestrator.strategies import StrategyEngine  # noqa: E402


class DemoProvider:
    """Deterministic provider for demonstrating orchestration workflows."""

    @property
    def provider_name(self) -> str:
        return "demo"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        timeout_seconds: float = 90.0,
    ) -> LLMResponse:
        system = messages[0].content.lower() if messages else ""
        user = messages[-1].content if messages else ""
        task = user.split("CONTEXT FROM PREVIOUS STEPS:")[0]
        task = task.replace("TASK:\n", "").replace("Task: ", "").strip()

        if "central supervisor" in system:
            used = user.lower()
            if "researcher" not in used:
                answer = {
                    "next": "researcher",
                    "reason": "Сначала необходимо собрать и структурировать информацию.",
                }
            elif "executor" not in used:
                answer = {
                    "next": "executor",
                    "reason": "После исследования можно подготовить решение.",
                }
            else:
                answer = {
                    "next": "finish",
                    "reason": "Основные этапы выполнены.",
                }

            import json
            content = json.dumps(answer, ensure_ascii=False)

        elif "planning agent" in system:
            content = (
                "План выполнения задачи:\n"
                "1. Определить цель и ограничения.\n"
                "2. Выделить необходимые этапы работы.\n"
                "3. Подготовить критерии качественного результата.\n"
                "4. Передать план следующим участникам."
                + self._route("researcher", "План готов, требуется исследование.")
            )

        elif "research agent" in system:
            content = (
                "Результаты исследования:\n"
                "1. REST API — интерфейс взаимодействия между приложениями "
                "по HTTP.\n"
                "2. Клиент отправляет запрос на сервер.\n"
                "3. Сервер возвращает ответ, обычно в формате JSON.\n"
                "4. Типичные методы: GET, POST, PUT и DELETE.\n"
                "Это общие сведения, а не результат обращения к внешним источникам."
                + self._route("executor", "Исследование завершено, можно готовить результат.")
            )

        elif "execution agent" in system:
            content = (
                "Подготовленный результат:\n\n"
                "REST API — способ взаимодействия программ через HTTP-запросы "
                "и ответы.\n\n"
                "Пример 1. GET /users — получение списка пользователей.\n"
                "Пример 2. POST /users — создание нового пользователя. "
                "Данные передаются в теле запроса, например в JSON.\n\n"
                "Таким образом, REST API позволяет разным приложениям "
                "обмениваться данными по согласованным правилам."
                + self._route("critic", "Черновик подготовлен и нуждается в проверке.")
            )

        elif "critical reviewer" in system:
            content = (
                "Проверка результата:\n"
                "1. Основное определение корректно для вводного объяснения.\n"
                "2. Примеры GET и POST соответствуют распространённому "
                "использованию HTTP.\n"
                "3. Следует уточнить, что REST — архитектурный стиль, "
                "а не отдельный протокол.\n"
                "4. Для учебного ответа достаточно добавить это уточнение."
            )

        elif "synthesis agent" in system:
            content = (
                "## Итоговый ответ\n\n"
                "REST API — интерфейс взаимодействия между приложениями, "
                "который обычно использует HTTP и следует принципам REST. "
                "REST является архитектурным стилем, а не отдельным протоколом.\n\n"
                "**Два примера:**\n\n"
                "- `GET /users` — получить список пользователей.\n"
                "- `POST /users` — создать пользователя, передав данные "
                "в теле запроса, например в JSON.\n\n"
                "Клиент отправляет запрос, сервер обрабатывает его и возвращает "
                "ответ с результатом."
            )

        else:
            content = "Демонстрационный ответ провайдера."

        return LLMResponse(
            content=content,
            model="demo-provider",
            prompt_tokens=None,
            completion_tokens=None,
            latency_seconds=0.01,
        )

    @staticmethod
    def _route(next_agent: str, reason: str) -> str:
        import json
        decision = {"next": next_agent, "reason": reason}
        return "\n" + json.dumps(decision, ensure_ascii=False)


STRATEGIES = {
    "Последовательная": "sequential",
    "Параллельная": "parallel",
    "Централизованная": "centralized",
    "Распределённая": "distributed",
}

AGENT_NAMES = {
    "planner": "Планировщик",
    "researcher": "Исследователь",
    "executor": "Исполнитель",
    "critic": "Критик",
    "synthesizer": "Синтезатор",
}


def run_strategy(
    strategy: str,
    task: str,
    provider_type: str,
    model: str,
    base_url: str,
    num_predict: int,
):
    if provider_type == "Ollama":
        provider = OllamaProvider(
            model=model,
            base_url=base_url,
            num_predict=num_predict,
            max_num_predict=4096,
            think=False,
        )
    else:
        provider = DemoProvider()

    engine = StrategyEngine(
        provider,
        timeout_seconds=180.0,
        max_steps=5,
        max_concurrency=2,
    )

    return asyncio.run(engine.run(strategy, task))


st.set_page_config(
    page_title="LLM Agent Orchestration Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("LLM Agent Orchestration Platform")
st.caption(
    "Платформа для запуска и исследования стратегий "
    "оркестрации LLM-агентов"
)

with st.sidebar:
    st.header("Настройки запуска")

    strategy_label = st.selectbox(
        "Стратегия оркестрации",
        list(STRATEGIES.keys()),
    )

    provider_type = st.selectbox(
        "Провайдер",
        ["Mock", "Ollama"],
    )

    model = "qwen3:1.7b"
    base_url = "http://localhost:11434"

    if provider_type == "Ollama":
        model = st.text_input("Модель", value=model)
        base_url = st.text_input("Адрес Ollama", value=base_url)
        num_predict = st.slider(
            "Максимум генерируемых токенов",
            min_value=256,
            max_value=4096,
            value=1024,
            step=256,
        )
    else:
        num_predict = 1024
        st.info(
            "Демонстрационный режим: ответы заранее подготовлены. "
            "Внешняя языковая модель не вызывается."
        )

    st.divider()
    st.caption("LLM Agent Orchestration Platform · v0.2.0")


st.subheader("Задача для команды агентов")

task = st.text_area(
    "Введите задачу",
    value=(
        "Кратко объясни, что такое REST API. "
        "Приведи два коротких примера."
    ),
    height=120,
)

if st.button(
    "▶ Запустить оркестрацию",
    type="primary",
    use_container_width=True,
):
    if not task.strip():
        st.warning("Введите задачу перед запуском.")
    else:
        try:
            with st.spinner("Команда агентов выполняет задачу..."):
                result = run_strategy(
                    strategy=STRATEGIES[strategy_label],
                    task=task,
                    provider_type=provider_type,
                    model=model,
                    base_url=base_url,
                    num_predict=num_predict,
                )

            st.session_state["last_result"] = result
            st.success("Оркестрация успешно завершена!")

        except Exception as exc:
            st.error(f"Ошибка выполнения: {exc}")


result = st.session_state.get("last_result")

if result is not None:
    st.divider()
    st.header("Результат выполнения")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Стратегия", result.strategy)
    metric2.metric(
        "Время выполнения",
        f"{result.duration_seconds:.2f} сек.",
    )
    metric3.metric("Количество агентов", len(result.agent_results))

    st.subheader("Итоговый ответ")
    st.markdown(result.final_answer)

    st.subheader("Ход выполнения агентов")

    for index, agent_result in enumerate(result.agent_results, start=1):
        agent_title = AGENT_NAMES.get(
            agent_result.agent_name,
            agent_result.agent_name,
        )

        with st.expander(
            f"{index}. {agent_title} · "
            f"{agent_result.latency_seconds:.2f} сек.",
            expanded=False,
        ):
            st.markdown(agent_result.content)

            col1, col2, col3 = st.columns(3)
            col1.metric("Модель", agent_result.model)
            col2.metric("Попытки", agent_result.attempts)

            tokens = (
                (agent_result.prompt_tokens or 0)
                + (agent_result.completion_tokens or 0)
                if agent_result.prompt_tokens is not None
                or agent_result.completion_tokens is not None
                else "Нет данных"
            )
            col3.metric("Токены", tokens)

    with st.expander("Техническая информация"):
        st.write("ID запуска:", result.run_id)
        st.write("Статус:", result.status)

        for event in result.events:
            st.write(
                event.timestamp.strftime("%H:%M:%S"),
                event.event_type.value,
                event.agent_name or "",
                event.message,
            )


# Comparative benchmark is available directly in Streamlit.
st.divider()
st.header("Сравнительный эксперимент")
st.caption(
    "Запуск одного набора задач всеми четырьмя стратегиями. "
    "Результаты отображаются здесь и дополнительно сохраняются в benchmark_results/."
)

with st.expander("Настройки эксперимента", expanded=True):
    bench_col1, bench_col2 = st.columns(2)
    benchmark_repetitions = bench_col1.number_input(
        "Повторов на каждую задачу и стратегию", min_value=1, max_value=10, value=3, step=1
    )
    benchmark_timeout = bench_col2.number_input(
        "Тайм-аут одного запуска, сек.", min_value=30, max_value=600, value=180, step=30
    )
    benchmark_dataset = st.text_input(
        "Файл набора задач", value=str(PROJECT_ROOT / "benchmarks" / "tasks.json")
    )
    st.caption(
        f"Будет выполнено до {len(BENCHMARK_STRATEGIES)} × "
        f"{int(benchmark_repetitions)} запусков на каждую задачу."
    )

if st.button("▶ Запустить сравнение четырёх стратегий", type="primary", use_container_width=True):
    try:
        tasks = load_tasks(Path(benchmark_dataset))
        if provider_type == "Ollama":
            benchmark_provider = OllamaProvider(
                model=model, base_url=base_url, num_predict=num_predict,
                max_num_predict=4096, think=False,
            )
        else:
            benchmark_provider = DemoProvider()

        total_runs = len(tasks) * len(BENCHMARK_STRATEGIES) * int(benchmark_repetitions)
        progress = st.progress(0, text="Подготовка эксперимента...")
        with st.spinner(f"Выполняется эксперимент: {total_runs} запусков..."):
            records = asyncio.run(
                run_benchmark(
                    benchmark_provider, tasks, repetitions=int(benchmark_repetitions),
                    timeout_seconds=float(benchmark_timeout),
                )
            )
        output_dir = PROJECT_ROOT / "benchmark_results"
        csv_path, json_path, summary_path = write_results(records, output_dir)
        st.session_state["benchmark_records"] = [r.__dict__ if hasattr(r, "__dict__") else {
            field: getattr(r, field) for field in r.__dataclass_fields__
        } for r in records]
        st.session_state["benchmark_summary"] = summarize(records)
        st.session_state["benchmark_files"] = {
            "csv": csv_path.read_bytes(), "json": json_path.read_bytes(),
            "summary": summary_path.read_bytes(),
        }
        st.session_state["benchmark_provider"] = provider_type
        progress.progress(100, text="Эксперимент завершён")
        st.success(f"Готово: {len(records)} запусков. Файлы сохранены в {output_dir}")
    except Exception as exc:
        st.error(f"Не удалось выполнить эксперимент: {exc}")

if "benchmark_summary" in st.session_state:
    st.subheader("Сводные результаты")
    st.caption(f"Провайдер: {st.session_state.get('benchmark_provider', '—')}")
    summary_rows = st.session_state["benchmark_summary"]
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    chart_rows = [
        {"Стратегия": row["strategy"], "Среднее время, сек.": row["mean_duration_seconds"]}
        for row in summary_rows if row["mean_duration_seconds"] is not None
    ]
    if chart_rows:
        st.subheader("Среднее время выполнения")
        st.bar_chart(chart_rows, x="Стратегия", y="Среднее время, сек.")

    st.subheader("Все запуски")
    st.dataframe(st.session_state["benchmark_records"], use_container_width=True, hide_index=True)
    download_col1, download_col2, download_col3 = st.columns(3)
    files = st.session_state["benchmark_files"]
    download_col1.download_button(
        "Скачать подробный CSV", data=files["csv"], file_name="benchmark_results.csv",
        mime="text/csv", use_container_width=True,
    )
    download_col2.download_button(
        "Скачать JSON", data=files["json"], file_name="benchmark_results.json",
        mime="application/json", use_container_width=True,
    )
    download_col3.download_button(
        "Скачать сводный CSV", data=files["summary"], file_name="benchmark_summary.csv",
        mime="text/csv", use_container_width=True,
    )
    st.info(
        "Важно: метрики времени и токенов являются операционными. "
        "Автоматическая оценка фактической правильности и полноты ответов не выполняется."
    )
