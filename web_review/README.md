# Local Review UI

Run from the repository root:

```bash
python3 web_review/server.py
```

Open <http://127.0.0.1:8775>. The composition tab covers all 112 audio files;
the QA tab covers all 253 questions. Both views read the canonical files under
`benchmark/` rather than copying the dataset.

Review state is saved to the two JSON files under `web_review/data/`. Commit
those files when annotations should move to another machine. The server binds
to localhost by default; use `--host 0.0.0.0` only behind an appropriate secure
tunnel or firewall.
