"""Real-time chat moderation demo (Gradio).

Every chat message is labeled clean / offensive with a confidence score by the selected model
(default: the distilled + INT8-quantized BERT-small, running on CPU with 1 thread). Flagged
messages are masked. The side panel logs the response time of every message against the
latency budget, and can optionally run all models side by side.

Run:  python app.py      -> open http://127.0.0.1:7860
"""
import re
import time
from datetime import datetime

import gradio as gr

from src.config import LATENCY_BUDGET_MS
from src.predictors import DEFAULT_MODEL, MODELS, load_model

_cache = {}


def get_model(name):
    if name not in _cache:
        _cache[name] = load_model(name)
        for t in EXAMPLES:  # warm-up, so the first real message is not measured cold
            _cache[name].predict_proba(t)
    return _cache[name]


def classify(name, text):
    model = get_model(name)
    t0 = time.perf_counter()
    p = model.predict_proba(text)  # includes regex normalization
    ms = (time.perf_counter() - t0) * 1000
    label = "offensive" if p >= 0.5 else "clean"
    conf = p if p >= 0.5 else 1 - p
    return label, conf, p, ms


def mask(text):
    """Masks a flagged message word by word, keeping only each word's first character
    ("shut up you idiot" -> "s*** u* y** i****"), so the chat layout stays readable."""
    return re.sub(r"\S+", lambda m: m.group(0)[0] + "*" * max(len(m.group(0)) - 1, 2), text)


def on_send(text, model_name, compare, history, log):
    text = (text or "").strip()
    if not text:
        return "", history, history, log, log, gr.update(), gr.update()
    label, conf, p, ms = classify(model_name, text)
    ok = ms <= LATENCY_BUDGET_MS
    if label == "offensive":
        shown = f"🚫 {mask(text)}\n\n`OFFENSIVE · {conf:.0%} · {ms:.1f} ms`"
    else:
        shown = f"{text}\n\n`clean · {conf:.0%} · {ms:.1f} ms`"
    history = history + [{"role": "user", "content": shown}]
    log = [[len(log) + 1, datetime.now().strftime("%H:%M:%S"), model_name, label, round(conf, 3), round(ms, 2),
            "✅" if ok else "❌"]] + log

    comparison = gr.update()
    if compare:
        rows = []
        for name in MODELS:
            l, c, pr, t = classify(name, text)
            rows.append([name, l, round(pr, 3), round(t, 2), "✅" if t <= LATENCY_BUDGET_MS else "❌"])
        comparison = gr.update(value=rows, visible=True)
    return "", history, history, log, log, comparison, stats_md(log)


def stats_md(log):
    if not log:
        return ""
    by_model = {}
    for row in log:
        by_model.setdefault(row[2], []).append(row[5])
    lines = [f"**Budget: {LATENCY_BUDGET_MS:.0f} ms / message (CPU, 1 thread)**", "", "| Model | Messages | Mean latency |", "|---|---|---|"]
    for m, ts in by_model.items():
        lines.append(f"| {m} | {len(ts)} | {sum(ts) / len(ts):.1f} ms |")
    return "\n".join(lines)


EXAMPLES = [
    "gg wp everyone, that last round was insane",
    "you are such a stuuupid n00b, uninstall",
    "shut the f@ck up you $hit player",
    "nobody wants you here, go back to where you came from",
    "that boss fight killed me lol",
    "@user your mom must be so proud of you, loser",
]

with gr.Blocks(title="Chat Moderation") as demo:
    gr.Markdown(
        "# 🛡️ Real-Time Chat Moderation\n"
        "Type a chat message. Each message is labeled **clean / offensive** live; offensive messages are masked. "
        f"Default model: **{DEFAULT_MODEL}**, on CPU, budget {LATENCY_BUDGET_MS:.0f} ms per message."
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Game chat", height=480)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message and press Enter…", show_label=False, scale=5)
                send = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=msg, label="Try these")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Moderation model")
            compare = gr.Checkbox(label="Also run every model on each message (side by side)", value=False)
            comparison = gr.Dataframe(headers=["Model", "Label", "P(offensive)", "Latency (ms)", "Budget"],
                                      label="All models, last message", visible=False, interactive=False)
            stats = gr.Markdown()
            log = gr.Dataframe(headers=["#", "Time", "Model", "Label", "Confidence", "Latency (ms)", "Budget"],
                               value=[], label="Response-time log", interactive=False)
            clear = gr.Button("Clear chat and log")
    hist_state, log_state = gr.State([]), gr.State([])

    inputs = [msg, model_dd, compare, hist_state, log_state]
    outputs = [msg, chatbot, hist_state, log, log_state, comparison, stats]
    msg.submit(on_send, inputs, outputs)
    send.click(on_send, inputs, outputs)
    clear.click(lambda: ([], [], [], [], gr.update(value=[], visible=False), ""), None,
                [chatbot, hist_state, log, log_state, comparison, stats])

if __name__ == "__main__":
    from src.config import STUDENT_INT8_PATH
    if not STUDENT_INT8_PATH.exists():  # first run on a fresh clone: fetch weights from the Hugging Face Hub
        from src.download import download
        download()
    print("Loading default model...")
    get_model(DEFAULT_MODEL)
    demo.launch()
