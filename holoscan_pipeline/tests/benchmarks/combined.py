"""Validation harness: one production PipelineApp, a warm-up and measured series."""
def main():
    from dataclasses import replace
    from pathlib import Path
    import json, threading, time
    from live4d.app import PipelineApp
    from live4d.config import load_config
    config = load_config(Path('/etc/live4d.yaml'))
    config = replace(config, execute=dict(mode='service', seconds=150))
    results = []
    def observe(batch, positioned):
        if batch.end_of_series:
            results.append(positioned.result)
    app = PipelineApp(config, observer=observe)
    Path('/tmp/live4d-plan.json').write_text(app.plan.json())
    stopping = threading.Event()
    def finish_when_drained():
        while not stopping.wait(.05):
            runtime = app.parallax_runtime
            if len(results) >= 2 and runtime.rounds >= 2 and not runtime.active:
                stopping.set()
                return
    threading.Thread(target=finish_when_drained, daemon=True).start()
    termination = app.run(stop_event=stopping)
    runtime = app.parallax_runtime
    print('[benchmark-complete] ' + json.dumps(dict(time=time.time(), termination=termination.name,
        series=len(results), bf=results[-1] if results else None,
        parallax_frames=runtime.sink.frames if runtime.sink else 0,
        parallax_complete=runtime.sink.completeness()[0] if runtime.sink else False)), flush=True)

if __name__ == "__main__":
    main()
    import gc
    gc.collect()
